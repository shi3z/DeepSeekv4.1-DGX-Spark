"""A browser chat front end for the DSV41 engine, on the box the engine runs on.

`server/app.py` binds 127.0.0.1, so a page loaded from anywhere else cannot reach it. This serves
the page and forwards /v1/* to the engine from the same origin, which also keeps the browser out
of CORS. Streaming responses are passed through chunk by chunk so a 19 tok/s model feels like it
is typing rather than hanging.

  python3 chatui.py --engine-port 8000 --port 8200 [--host 0.0.0.0]
"""
from __future__ import annotations

import argparse
import http.server
import json
import socketserver
import sys
import urllib.error
import urllib.request

PAGE = r"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DeepSeek V4.1 Flash on one DGX Spark</title>
<style>
:root{--bg:#faf9f7;--fg:#1a1a1a;--mut:#6b6b6b;--line:#e3e0db;--me:#eef2ff;--card:#fff;--acc:#2f6feb;--ok:#127a3d}
@media (prefers-color-scheme:dark){:root{--bg:#16171a;--fg:#e8e6e3;--mut:#9a9a9a;--line:#2c2e33;--me:#1e2a44;--card:#1d1f24;--acc:#6f9bff;--ok:#4ec77f}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Hiragino Sans","Noto Sans JP",sans-serif}
header{border-bottom:1px solid var(--line);padding:14px 16px}
.hwrap{max-width:880px;margin:0 auto}
h1{font-size:16px;margin:0 0 6px;font-weight:650}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;background:var(--acc);color:#fff;font-size:11px;font-weight:700;letter-spacing:.03em;vertical-align:2px;margin-right:8px}
.hw{font-size:12px;color:var(--mut);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;line-height:1.9}
.hw b{color:var(--fg);font-weight:600}
main{max-width:880px;margin:0 auto;padding:16px}
.msg{margin:0 0 14px;padding:11px 14px;border-radius:10px;background:var(--card);border:1px solid var(--line);white-space:pre-wrap;word-wrap:break-word;overflow-wrap:anywhere}
.msg.user{background:var(--me)}
.who{font-size:11px;color:var(--mut);letter-spacing:.04em;text-transform:uppercase;margin-bottom:5px}
.stats{margin-top:9px;padding-top:8px;border-top:1px dashed var(--line);font-size:12px;color:var(--mut);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;display:flex;gap:16px;flex-wrap:wrap;align-items:baseline}
.tps{font-size:17px;font-weight:700;color:var(--ok)}
form{max-width:880px;margin:0 auto;padding:0 16px 24px;display:flex;gap:8px;align-items:flex-end}
textarea{flex:1;min-height:52px;max-height:40vh;padding:10px 12px;border:1px solid var(--line);border-radius:10px;background:var(--card);color:var(--fg);font:inherit;resize:vertical}
button{padding:11px 18px;border:0;border-radius:10px;background:var(--acc);color:#fff;font:inherit;font-weight:600;cursor:pointer}
button:disabled{opacity:.45;cursor:default}
.row{max-width:880px;margin:0 auto;padding:0 16px 10px;font-size:12px;color:var(--mut);display:flex;gap:14px;align-items:center;flex-wrap:wrap}
input[type=number]{width:76px;padding:4px 6px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg);font:inherit}
</style></head><body>
<header><div class="hwrap">
  <h1><span class="badge">1&times; DGX SPARK</span>DeepSeek V4.1 Flash</h1>
  <div class="hw">
    <b>hardware</b> NVIDIA GB10 &times;1 &middot; 121 GiB unified memory &middot; 20&times; Arm Cortex-X925 &middot; single node, no tensor parallelism<br>
    <b>model</b> <span id="mdl">deepseek-v4.1-flash</span> &middot; <span id="cfg">connecting&hellip;</span>
  </div>
</div></header>
<main id="log"></main>
<div class="row">
  <label>max tokens <input type="number" id="mx" value="400" min="16" max="4000"></label>
  <label>temperature <input type="number" id="tp" value="0.7" min="0" max="2" step="0.1"></label>
  <label><input type="checkbox" id="keep" checked> 会話を継続</label>
  <span id="agg"></span>
  <button type="button" id="clear" style="background:transparent;color:var(--mut);padding:4px 8px">履歴を消す</button>
</div>
<form id="f"><textarea id="q" placeholder="メッセージを入力（Ctrl+Enter で送信）"></textarea><button id="go">送信</button></form>
<script>
const log=document.getElementById('log'),q=document.getElementById('q'),go=document.getElementById('go');
let history=[],sumTok=0,sumSec=0;
function add(who,text,cls){const d=document.createElement('div');d.className='msg '+(cls||'');
  const w=document.createElement('div');w.className='who';w.textContent=who;
  const b=document.createElement('div');b.textContent=text;d.append(w,b);log.append(d);
  window.scrollTo(0,document.body.scrollHeight);return b;}
function cfgLine(s){
  if(!s)return;
  const ex=s.expert_format==='cb3'?'CB3 3-bit codebook experts ('+s.expert_mb+' MB each)':s.expert_format+' experts';
  const parts=[ex];
  if(s.dense_fp4&&s.dense_fp4!=='off')parts.push('dense FP4: '+s.dense_fp4);
  if(s.head_fmt)parts.push('LM head '+s.head_fmt.toUpperCase());
  if(s.arena_slots)parts.push(s.arena_slots.toLocaleString()+' experts resident in a '+
     Math.round((s.arena_slots*(s.expert_mb||14.45))/1024)+' GB arena, the rest streamed from NVMe');
  document.getElementById('cfg').textContent=parts.join(' · ');
}
document.getElementById('clear').onclick=()=>{history=[];log.innerHTML='';};
q.addEventListener('keydown',e=>{if(e.key==='Enter'&&(e.ctrlKey||e.metaKey))document.getElementById('f').requestSubmit();});
document.getElementById('f').onsubmit=async e=>{
  e.preventDefault();const text=q.value.trim();if(!text)return;
  q.value='';go.disabled=true;add('you',text,'user');
  const msgs=document.getElementById('keep').checked?history.concat([{role:'user',content:text}]):[{role:'user',content:text}];
  const body={messages:msgs,max_tokens:+document.getElementById('mx').value,
              temperature:+document.getElementById('tp').value,stream:true};
  const out=add('deepseek-v4.1-flash','');
  const t0=performance.now();let acc='',st=null;
  try{
    const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});
    if(!r.ok){out.textContent='HTTP '+r.status+' '+(await r.text()).slice(0,400);go.disabled=false;return;}
    const rd=r.body.getReader(),dec=new TextDecoder();let buf='';
    for(;;){const{done,value}=await rd.read();if(done)break;
      buf+=dec.decode(value,{stream:true});
      let i;while((i=buf.indexOf('\n'))>=0){
        const line=buf.slice(0,i).trim();buf=buf.slice(i+1);
        if(!line.startsWith('data:'))continue;
        const p=line.slice(5).trim();if(p==='[DONE]')continue;
        let j;try{j=JSON.parse(p)}catch(_){continue}
        if(j.error){out.textContent=acc+'\n\n[engine error: '+(j.error.message||JSON.stringify(j.error))+']';continue}
        if(j.x_engine_stats){st=j.x_engine_stats;cfgLine(st);}
        const d=j.choices&&j.choices[0]&&(j.choices[0].delta||j.choices[0].message);
        if(d&&d.content){acc+=d.content;out.textContent=acc;window.scrollTo(0,document.body.scrollHeight);}
      }}
  }catch(err){out.textContent=acc+'\n\n[stream error: '+err+']';}
  const wall=(performance.now()-t0)/1000;
  const s=document.createElement('div');s.className='stats';
  const tps=st&&st.decode_tok_s?st.decode_tok_s:null;
  const n=st&&st.completion_tokens?st.completion_tokens:null;
  s.innerHTML='<span class="tps">'+(tps!==null?tps.toFixed(2)+' tok/s':'—')+'</span>';
  const bits=[];
  if(n)bits.push(n+' tokens');
  if(st&&st.decode_s)bits.push('decode '+st.decode_s.toFixed(2)+' s');
  if(st&&st.prefill_s)bits.push('prefill '+st.prefill_s.toFixed(2)+' s ('+(st.prefill_tok_s||0).toFixed(0)+' tok/s)');
  if(st&&st.accept_len_mean)bits.push('DSpark accept '+st.accept_len_mean.toFixed(2));
  if(st&&st.expert_hit_rate!==undefined)bits.push('expert cache hit '+(st.expert_hit_rate*100).toFixed(1)+'%');
  if(st&&st.nvme_gb)bits.push('NVMe '+st.nvme_gb.toFixed(2)+' GB');
  bits.push('wall '+wall.toFixed(1)+' s');
  for(const b of bits){const e=document.createElement('span');e.textContent=b;s.append(e);}
  out.parentElement.append(s);
  if(tps!==null&&n){sumTok+=n;sumSec+=(st.decode_s||0);
    document.getElementById('agg').textContent='session: '+sumTok+' tokens @ '+(sumTok/Math.max(sumSec,1e-9)).toFixed(2)+' tok/s';}
  if(document.getElementById('keep').checked){history=msgs.concat([{role:'assistant',content:acc}]);}
  go.disabled=false;q.focus();
};
fetch('/v1/models').then(r=>r.json()).then(j=>{
  const m=j.data&&j.data[0];if(m){document.getElementById('mdl').textContent=m.id+' (ctx '+m.max_model_len+')';}
  document.getElementById('cfg').textContent='送信すると構成と tok/s が表示されます';
}).catch(()=>{document.getElementById('cfg').textContent='engine unreachable';});
</script></body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    engine = "http://127.0.0.1:8000"

    def log_message(self, fmt, *a):          # one line per request, not three
        sys.stderr.write("%s %s\n" % (self.command, self.path))

    def _page(self):
        b = PAGE.encode()
        self.send_response(200)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _err(self, code, payload):
        b = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _proxy(self, body=None):
        """Relay to the engine and stream the body back.

        The body is delimited by the connection close, not by a length or by chunks this handler
        writes itself: the first version framed the stream by hand and, when anything went wrong
        mid-stream, tried to send a fresh 502 after the headers were already out. That corrupted
        the framing and the browser stopped reading -- which looked exactly like the model
        truncating its answer after two tokens.
        """
        req = urllib.request.Request(self.engine + self.path, data=body, method=self.command,
                                     headers={"content-type": "application/json"})
        try:
            r = urllib.request.urlopen(req, timeout=1800)
        except urllib.error.HTTPError as e:
            return self._err(e.code, e.read())
        except Exception as e:
            return self._err(502, {"error": str(e)})
        self.send_response(r.status)
        self.send_header("content-type", r.headers.get("content-type", "application/json"))
        self.send_header("cache-control", "no-store")
        self.send_header("connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            while True:
                chunk = r.read1(65536)     # returns as soon as anything is there, unlike read()
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except Exception as e:             # the client went away, or the engine did; never
            sys.stderr.write(f"stream ended: {e}\n")   # try to write a new response here
        finally:
            r.close()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._page()
        return self._proxy()

    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        return self._proxy(self.rfile.read(n))


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-port", type=int, default=8000)
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    Handler.engine = f"http://127.0.0.1:{a.engine_port}"
    print(f"chat UI  http://{a.host}:{a.port}/   ->  engine {Handler.engine}", flush=True)
    Server((a.host, a.port), Handler).serve_forever()

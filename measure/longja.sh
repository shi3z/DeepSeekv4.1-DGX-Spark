#!/bin/bash
# One long Japanese generation: the case where a wrong resident set costs the most, and where the
# earlier failures showed up (short answers survived everything).
timeout 400 curl -s http://127.0.0.1:8100/v1/chat/completions -H 'content-type: application/json' \
 -d '{"messages":[{"role":"user","content":"日本の四季について、それぞれの季節の気候と代表的な行事を交えて400字程度で説明してください。"}],"max_tokens":400,"temperature":0.0}' \
 | python3 -c "
import json,sys
d=json.load(sys.stdin); st=d.get('x_engine_stats',{})
t=d['choices'][0]['message']['content']
print('chars', len(t), '| distinct-char ratio', round(len(set(t))/max(1,len(t)),3))
print(t[:200].replace(chr(10),' '))
print('STATS', {k:st.get(k) for k in ('decode_tok_s','accept_len_mean','expert_hit_rate','nvme_gb_per_token','prefill_s')})
"

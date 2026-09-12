#!/bin/bash
# The same small battery against whatever is on :8100, so two configurations can be compared on
# identical prompts. Factual recall and language consistency are what the first failure showed.
Q1='日本で一番高い山と、その標高を教えてください。'
Q2='フランスの首都はどこですか。一文で答えてください。'
Q3='徳川家康が江戸幕府を開いた年は？年号だけ答えてください。'
Q4='Write a Python function that reverses a string. Code only.'
for q in "$Q1" "$Q2" "$Q3" "$Q4"; do
  echo "### $q"
  timeout 200 curl -s http://127.0.0.1:8100/v1/chat/completions -H 'content-type: application/json' \
    -d "$(python3 -c "import json,sys;print(json.dumps({'messages':[{'role':'user','content':sys.argv[1]}],'max_tokens':110,'temperature':0.0}))" "$q")" \
  | python3 -c "
import json,sys
d=json.load(sys.stdin)
if 'choices' not in d: print('ERROR', str(d)[:200]); raise SystemExit
st=d.get('x_engine_stats',{})
print(d['choices'][0]['message']['content'][:320].replace(chr(10),' | '))
print('    ', {k:st.get(k) for k in ('decode_tok_s','accept_len_mean')})
"
done

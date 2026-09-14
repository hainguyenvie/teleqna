import json, random, concurrent.futures, urllib.request, os, sys, time
sys.path.insert(0,'infra/teleqna'); from run_baseline import build_sample, build_prompt, parse_answer
URL="https://stream-netmind.viettel.vn/aigw/ai/v1/chat/completions"; KEY=os.environ["GATEWAY_KEY"]
test={json.loads(l)['sample_id']:json.loads(l) for l in open('data/teleqna/test.jsonl')}
rs=[json.loads(l) for l in open('artifacts/teleqna-gateway/otfull_think/results.jsonl')]
rs=[r for r in rs if r['reasoning'] and r['parsed_answer']]
gold=lambda i: chr(65+test[i]['answer'])
random.seed(7)
wrong=random.sample([r for r in rs if not r['correct']],200); right=random.sample([r for r in rs if r['correct']],200)
P_ADV=("Below is a multiple choice question and a draft reasoning trace written by another analyst. "
 "The draft may be wrong: these questions are generated from specific 3GPP specifications and research papers, "
 "and the correct option is the one that matches the SOURCE TEXT's own wording and claims, not the most plausible "
 "general-knowledge answer. Check each option the draft considered and rejected: was it rejected for a reason the "
 "source would agree with, or because it sounded less generic? Decide the option the source text itself would state.\n\n"
 "{q}\n\n--- DRAFT REASONING ---\n{tr}\n--- END ---\n\nEnd your response with the line 'ANSWER: $LETTER'.")
def call(r):
    s=build_sample(test[r['sample_id']],False); q=build_prompt(s,False)
    payload={"model":"Qwen/Qwen3.5-122B-A10B-FP8","messages":[{"role":"user","content":P_ADV.format(q=q,tr=r['reasoning'][:12000])}],"temperature":0,"max_tokens":16000,"seed":42,"enable_thinking":True}
    for a in range(5):
        try:
            req=urllib.request.Request(URL,data=json.dumps(payload).encode(),headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
            d=json.load(urllib.request.urlopen(req,timeout=1500)); return parse_answer(d['choices'][0]['message'].get('content') or '', len(s['choices']))
        except Exception: time.sleep(3*(a+1))
    return ''
with concurrent.futures.ThreadPoolExecutor(24) as ex:
    fw=list(ex.map(call,wrong)); fr=list(ex.map(call,right))
fix=sum(a==gold(r['sample_id']) for a,r in zip(fw,wrong)); brk=sum(a!=gold(r['sample_id']) for a,r in zip(fr,right))
pw=1574/7256; print(f"adversarial-think fix {fix}/200 break {brk}/200 keeps-own-wrong {sum(a==r['parsed_answer'] for a,r in zip(fw,wrong))} unparsed {sum(a=='' for a in fw+fr)} => est net {100*(pw*fix/200-(1-pw)*brk/200):+.2f}pp", flush=True)
json.dump({"wrong":[(r['sample_id'],a) for a,r in zip(fw,wrong)],"right":[(r['sample_id'],a) for a,r in zip(fr,right)]},open('artifacts/teleqna-gateway/verify_pilot_think.json','w'))

# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
from flask_socketio import SocketIO, send, emit
from flask_cors import CORS
#from root import redirect_test
import json
import os

from termcolor import colored
import random
from dotenv import load_dotenv

from APIService import *
from character_interaction import *
from game_setting import *


temp = [
{"dialogues": "늑대와 빨간 망토의 이야기는 정말 흥미롭게 들립니다. 빨간 망토가 늑대에게 복수하러 가기로 결정한 이유는 무엇일까요? 늑대에 대한 제 생각은 그렇게 좋지 않습니다. 하지만, 이야기를 더 알아가면서 두 캐릭터의 동기와 내면을 이해할 수 있을 것 같습니다. 복수가 정말 필요한 일인지에 대해서는 단면적인 판단보다는 이야기의 전개와 캐릭터의 성장을 통해 판단해야 할 것 같습니다. 여러분은 어떻게 생각하시나요? 늑대와 빨간 망토의 이야기는 정말로 흥미로워 보입니다. 빨간 망토가 늑대에게 복수하려는 이유는 아마도 어떤 상처나 배신을 받았기 때문일 수도 있을 것 같습니다. 늑대에 대한 생각이 좋지 않다는 것은 이야기 속에서 늑대가 어떤 행동을 했거나 어떤 인상을 주었기 때문일 것입니다. 하지만, 이야기를 더 알아가면서 두 캐릭터의 동기와 내면을 이해하고 그들의 성장을 지켜보는 것이 중요하다고 생각합니다. 복수가 정말 필요한 일인지 판단하기 위해서는 이야기의 전개를 따라가야 하며, 캐릭터들이 어떻게 변화하고 성장하는지 관찰해야 할 것 같습니다. 제 생각에는 이야기의 흐름과 캐릭터의 변화를 통해 판단하는 것이 단면적인 판단보다 더 의미 있는 결론을 도출할 수 있을 것 같습니다. 여러분은 어떻게 생각하시나요? 그 질문에 대해서 생각하는데 시간이 좀 필요할 것 같아요. 좀 더 생각한 뒤에 답변을 드릴게요." },
{"dialogues": "당신과 늑대 사이에는 무슨 일이 있었나요? 늑대에 대해 어떻게 생각해요? 그리고 wildcat이(가) 빨간 망토와 함께 복수하러 가기로 결정한 것이 잘된 일일까요? 아마도 빨간 망토는 늑대에게 복수를 하려면 어떤 계획을 세워야 할 것 같아요. 그녀가 어떤 능력이나 무기를 가지고 있는지 알아야 해요. 맞아요, 늑대가 왜 인간을 위협하는지에 대해서도 알아봐야겠네요. 우리는 상황을 파악하고 빨간 망토가 어떤 행동을 취할지 고려해야 해요. 네, 우리가 그녀를 찾아보고 그녀의 행동을 조사해야겠어요. 그렇게 하면 그녀가 왜 인간들을 위협하는지 알아낼 수 있을 거에요. 그리고 그녀의 동기와 계획을 파악해서 그녀를 막을 수 있을 거에요. 어떻게 시작할까? 그녀를 추적하기 전에 먼저 우리가 어떤 정보를 가지고 있는지 확인해야겠어요. 그녀의 행동 패턴, 목표, 그리고 그녀를 인간들에게 위협으로 만드는 요인에 대해 알아보는 게 중요하니까요. 그래야 그녀를 막을 수 있는 최선의 전략을 세울 수 있을 거에요."},
{"dialogues": "늑대가 빨간 망토의 할머니를 잡아먹으려고 했다는 소문 들었어요? mouse이(가) 빨간 망토와 함께 복수하러 가기로 결정한 것이 잘된 일일까요? 우리 같이 토론해봐요. 늑대는 사람들에게 두려움을 주는 동시에 동물들에게도 위협이 되는 존재라고 생각해요. 그래도 어떤 사람들은 늑대를 잡아먹는 괴물로 여기지만, 실은 늑대도 사람들을 도우려는 마음을 가지고 있는걸요. 알아요, 늑대와 함께 모험을 하면서 서로를 이해하고 친해지는 이야기들도 있어요. 그리고 용감한 영웅들의 이야기도 많아요.그들의 영웅적인 모습과 모험을 담은 이야기들도 있어요. 그래서 저는 늑대와 빨간망토가 서로를 이해하고 친해지는 소식을 듣고 싶어요."}
]
temps = 3

#from server import APIServer

load_dotenv()



####################################### 배포 시 주석 풀 것
""" 본인 OpenAI API Key 삽입 """
openai.api_key = os.getenv('OPENAI_API_KEY') #json.loads(get_secret()).get("OPENAI_API_KEY")
# C:\Users\komj\Desktop\Afore\afore\Lib\site-packages\openai\__init__.py



""" Flask APP 생성 """
app = Flask(__name__)
CORS(app, resources={r'*': {'origins': '*'}})
app.config['JSON_AS_ASCII'] = False
#app.config['SECRET_KEY'] = 'abc'
#CORS(app) # 혹은 허용할 도메인 지정할 경우 -> CORS(app, resources={r'*': {'origins': 'https://www.naver.com'}})
socketio = SocketIO(app) #async_mode='gevent')


""" API Service methods 매칭 
def is_api_function(obj):
    return inspect.isfunction(obj) and not obj.__name__.startswith('_')

api_functions_dict = {
    name: func
    for name, func in inspect.getmembers(APIServer, is_api_function)
}
"""


""" ROUTE """

@app.route('/')
def start():
    return "hello world"

##################################################### -> CS 수정
@app.route('/fairy-tale/api')
def index():
    return "fairy-tale/api - OPENAI_API_KEY" #render_template('index.html')


""" NPC-NPC 상호작용 """

@app.route('/fairy-tale/npc2npc/interviewNPC', methods=['POST'])
def interview_npc():
    # { 'npc_name' = "mouse|wildcat|bear", 'pursuade_tf' : true|false }

    # user와 대화한 npc 이름 및 늑대 복수 설득 여부
    response = request.get_json()
    npc_name = response.get('npc_name')
    pursuade = response.get('pursuade_tf')

    
    # npc 생성
    npcs = []
    memories = []
    #observations = []

    name1, traits1 = _create_random_npc()
    name2, traits2 = _create_random_npc()

##    n, m = get_agent(name1, 36, traits1, "He is talking with " + name2)
##    npcs.append(n); memories.append(m) 
##    n, m = get_agent(name2, 36, traits2, "He is talking with " + name1)
##    npcs.append(n); memories.append(m) 

    # NPC간 대화 상황설정 프롬프트 : NPC 설득 여부 반영한 인터뷰 형식(이 함수에서는 두 개만 반영)
    npcs, observations = _set_interview(npc_name, npcs, pursuade)

    """
    # 매 관찰마다 시스템이 생성하는 요약문을 확인, 요약문 발전 감시(for 성능 평가 및 개선)
    print("-------------------- 각 NPC의 성격과 상황 --------------------")
    for o in range(len(observations)):
        for i, observation in enumerate(observations[o]):
            _, reaction = npcs[o].generate_dialogues_response(observation)  # bool, str
            print(colored(observation, "green"), reaction)
            #npcs[o].get_summary(force_refresh=True)
            #print(colored( f"After {i + 1} observations, {npcs[o].name}'s summary is:\n{npcs[o].get_summary(force_refresh=True)}", "blue",))
            """ 

    # NPC간 대화
    print("-------------------- NPC 간의 대화 --------------------")
    
    if pursuade : pin2 = f"늑대가 빨간 망토의 할머니를 잡아먹으려고 했다는 소문 들었어요? 빨간 망토는 너무 화가 나서 늑대에게 복수하러 간대요. 당신과 늑대 사이에는 무슨 일이 있었나요? 늑대에 대해 어떻게 생각해요? 그리고 {npc_name}이(가) 빨간 망토와 함께 복수하러 가기로 결정한 것이 잘된 일일까요? 우리 같이 토론해봐요.",
    else : pin2 = f"늑대가 빨간 망토의 할머니를 잡아먹으려고 했다는 소문 들었어요? 빨간 망토는 너무 화가 나서 늑대에게 복수하러 간대요. 당신과 늑대 사이에는 무슨 일이 있었나요? 늑대에 대해 어떻게 생각해요? 그리고 {npc_name}이(가) 복수하러 가지 않기로 결정한 것이 잘된 일일까요? 우리 같이 토론해봐요.",

##    observation = run_conversation(
##        npcs,
##        pin2
##    )

    print("이제 시작")
    
    print("content :....")
    
    # npc간 대화 요약
    content = ""
    for p in public:
        content += p

    content += "I want to summarize this conversation in detail within 1000 tokens. Can you summarize it in Korean? Write : 'what to say'"
    print(content)
##    completion = create_gpt_saying([{"role": "user", "content": content}])
##    answer = completion["choices"][0]["message"]["content"].strip()
##   public_dialogues_summary.append(answer)
    
    print("public_dialogues_summary")
    print(public_dialogues_summary)
    print("public_dialogues")
    print(public_dialogues)
    print("public")
    print(public)

### temp 입니다 
    temps -= 1
    answer = temp[temps]
    if temps == 0:
        temps = 3
    

    
    #return json.dumps({"dialogues": public_dialogues}, ensure_ascii=False) # 수정? public_dialogues + npcs'name / public
    return json.dumps({"dialogues": answer}, ensure_ascii=False) # 수정? public_dialogues + npcs'name / public


""" notion : api 명세서에 작성하기
@application.route('/fairy-tale/npc2npc/dialogue', methods=['GET'])
def get_dialogues():
    # npc-npc 대화 함수를 npc-user 'exit' 입력 시 호출
    # 이 함수를 6초에 한 번씩 호출
    global public_dialogues
    response = public_dialogues
    public_dialogues = []
    return json.dumps({"dialogues": response}, ensure_ascii=False)
"""

def _create_random_npc():
    # 이름
    fm = random.randint(0, 1)
    if fm == 0 : name = names_male[random.randint(0, len(names_male)-1)]
    else : name = names_female[random.randint(0, len(names_female)-1)]

    # 성격
    traits = ""
    for _ in range(2):
        personality = personalities[random.randint(0, len(personalities)-1)]
        traits += personality
        traits += ", "
    
    return name, traits
    

def _set_interview(npc_name, npcs, pursuade):
    observation = []

    if pursuade : pin = f"빨간 망토는 자신의 할머니를 잡아먹으려고 한 늑대에게 복수하기 위해 {npc_name}(을)를 설득했고, 마침내 {npc_name}은(는) 함께 복수하러 가기로 했다는 소식을 들었다. {npcs[1].name}은(는) {npc_name}의 결정을 어떻게 생각하는지 물어보았다. 이에 대해 토론한다."
    else : pin = f"빨간 망토는 자신의 할머니를 잡아먹으려고 한 늑대에게 복수하기 위해 {npc_name}(을)를 설득했지만, {npc_name}은(는) 복수하러 가지 않기로 했다는 소식을 들었다. {npcs[1].name}은(는) {npc_name}의 결정을 생각하는지 물어보았다. 이에 대해 토론한다."

    for i in range(2):
        exp = random.randint(0, 1)
        if exp == 0: context = negative_experiences[random.randint(0, len(negative_experiences)-1)]
        else : context = positive_experiences[random.randint(0, len(positive_experiences)-1)]

        if i == 0:
            npc_obsv = [f"'" + context + "'", f"'{pin}'"]
            observation.append(f"{npcs[i].name} said to {npcs[i+1].name} " + npc_obsv[0])
            observation.append(f"{npcs[i].name} said to {npcs[i+1].name} " + npc_obsv[1])
        else : 
            npc_obsv = [f"'" + context + "'", f"'빨간 망토는 자신의 할머니를 잡아먹으려고 한 늑대에게 복수하기 위해 {npc_name}(을)를 설득했지만, {npc_name}은(는) 복수하러 가지 않기로 했다는 소식을 들었다.'"]
            observation.append(f"{npcs[i].name} said to {npcs[i-1].name} " + npc_obsv[0])
            observation.append(f"{npcs[i].name} said to {npcs[i-1].name} " + npc_obsv[1])

    for o in observation:
                npcs[0].memory.add_memory(o)
                npcs[1].memory.add_memory(o)
    
    return npcs, observation


""" User-NPC 상호 작용 """

@app.route('/fairy-tale/user2npc/start', methods=['POST'])
def start_to_talk():
    # {"npc_name" : "mouse|wildcat|bear"}
    response = request.get_json()
    name = response.get("npc_name")

    start_to_talking(name)

    return json.dumps({"npc_name": f"{name}"}, ensure_ascii=False)

@app.route('/fairy-tale/user2npc/talkToNPC', methods=['POST'])
def talk_to_npc():
    # 대화의 시작(NPC) : {"role": "assistant", "content": "빨간 망토야, 너의 할머니가 늑대에게 잡아먹힐 뻔 했다는 얘기를 들었어. 정말 화가 많이 났겠다."}
    # {"npc_name" : "mouse|wildcat|bear", "content": "ex) 나는 너무 화가 나. 너는 어때? | exit" }
    response = request.get_json()

    name = response.get('npc_name')
    user_speaking = response.get('content')
    completion = talk_with_npc(name, user_speaking)
    return json.dumps(completion, ensure_ascii=False)  # NPC의 응답 JSON 변환
    # {
    #     "prompt_tokens": 509,
    #     "total_tokens": 633
    #     "role": "assistant",
    #     "content": "나도 공감해. 예전에 늑대가 내 그림을 망가뜨렸을 때 정말 화가 났었어. 지금은 화가 약 30% 정도인 것 같아. | True | False"
    #     "message": [{"role": "system", "content": "You're a helpful assistant."}, {"role": "user", "content": "I want chatgpt to make his own decisions based on the context and make a story."}, {"role": "assistant", "content": "I will no longer provide specific dialogue guidelines, so please allow ChatGPT to proceed freely.This will allow ChatGPT to leverage the context given to create creative conversations.From now on, I will not limit my response and let ChatGPT continue the conversation freely."},]
    # }

# 지금까지 대화 내용 및 대화 횟수(서버) 확인
@app.route('/fairy-tale/user2npc/record', methods=['GET'])
def get_dialogue():
    print("대화 수 : " + str(len(messages)-8)) # 프롬프트 구성 제외
    return json.dumps(messages, ensure_ascii=False)
    """
    # 지금까지의 대화 내용 확인
    ...
    {"content":"But you have no intention of following me to avenge the wolf. Never. Your anger towards the wolf is about 10%, and my anger towards the wolf is about 80%. You can follow me after hearing my persuasion to avenge the wolf. And you might not follow me. After about 5 conversations, you'll decide for yourself. Must be creative in dialogue. You'll go first. Say all the answers in Korean.","role":"user"},
    {"content":"\ub098\ub294 \ub108\ubb34 \ud654\uac00 \ub098. \ub108\ub294 \uc5b4\ub54c?","role":"user"}
    ...
    """

# NPC별 대화 내용 요약 및 설득 여부
@app.route('/fairy-tale/user2npc/summary', methods=['GET'])
def summarize_dialogue():

### temp 입니다 

    return json.dumps({"pursuade":
                           [{"mouse": "한때 마을에 살던 빨간 망토는 어느 날, 할머니의 안부를 전하러 가는 도중 늑대가 그녀를 위협하는 걸 목격한다. 분노와 원한으로 가득 찬 망토는 늑대에게 복수하기로 결심한다. 하지만 혼자서 갈 수 없다고 생각해 함께 가야할 동료인 지혜로운 쥐와 함께 계획을 세우기로 한다. 그렇게 시작된 망토와 쥐의 복수 여정은 도전과 위험이 가득하지만, 이들은 서로를 힘들 때마다 격려하며 함께 극복해나가는 용기 있는 친구가 되어간다. 이제 그들은 과연 늑대에게 정의를 가져올 수 있을까?", "pursuaded_tf": True}, 
                           {"wildcat": "한때 빨간 망토와 늑대는 함께 있었지만, 음식 취향의 차이로 인해 헤어졌다. 어느 날, 빨간 망토는 할머니를 위협하는 늑대를 목격하고 분노에 가득 찼다. 그래서 복수하기 위해 동료인 야생 고양이를 설득하려 했다. 처음에 야생 고양이는 복수에 관심이 없었지만, 망토의 열정과 할머니의 안전을 생각하면서 결국 동참하기로 했다. 이제 그들은 함께 복수 계획을 세우고 실천할 준비가 되어있다.", "pursuaded_tf": True},
                           {"bear": "한때 활기차고 강한 청년이었던 빨간 망토는 어린 시절 늑대와의 전투에서 패배하며 상처를 입었다. 그로 인해 지금은 우울함과 두려움에 사로잡혀 있었다. 어느 날, 할머니가 늑대에게 잡아먹힐 뻔한 사건을 듣고 분노를 느낀다. 이번 기회에 복수하기 위해 결심을 한다. 하지만 너그러운 망토는 자신의 상처와 가족을 걱정하여 주저하는 모습이 보인다. 그래도 근거 없는 자신의 분노를 이해하지 않으며, 복수를 통해 망토의 상처가 치유될 수 있다고 설득한다. 겁이 나는 만큼 큰 일이라 생각하며, 망토와 함께 움직일 준비가 되어있다면 대답한다.", "pursuaded_tf": True}]
                        }, ensure_ascii=False)

##    return json.dumps({"pursuade":
##                           [{"mouse": user2npc_summary[0], "pursuaded_tf": pursuaded[0]},  # response['pursuade'][0]['mouse'] (문자열 값) | response['summary'][0]['pursuaded_tf'] (boolean 값)
##                           {"wildcat": user2npc_summary[1], "pursuaded_tf": pursuaded[1]}, # response['pursuade'][1]['wildcat'] (문자열 값) | response['summary'][1]['pursuaded_tf'] (boolean 값)
##                           {"bear": user2npc_summary[2], "pursuaded_tf": pursuaded[2]}]    # response['pursuade'][2]['bear'] (문자열 값) | response['summary'][2]['pursuaded_tf'] (boolean 값)
##                        }, ensure_ascii=False)             


# 최종 게임 결과 전달받기
@app.route('/fairy-tale/final/game', methods=['POST'])
def receive_game_result():
    # {"game_result" : true|false} # 승|패
    response = request.get_json()
    game_result = response.get("game_result")
    global final_game
    final_game = game_result

    return json.dumps({"game_result": game_result}, ensure_ascii=False)  # echo

# 최종 게임 결과 echo
@app.route('/fairy-tale/final/game', methods=['GET'])
def send_game_result():
    return json.dumps({"game_result": final_game}, ensure_ascii=False)


# 최종 스토리 요약 생성
# 최종 스토리 요약 생성
@app.route('/fairy-tale/final/story', methods=['GET'])
def create_user_story():
    global user2npc_summary, pursuaded, final_game, public_dialogues_summary, story, public, public_dialogues, messages, npcs, memories, observations
    msgs = ""
    npc_pursuaded = ""

    # user2npc / npc2npc 요약 / 최종 게임 결과
    for i in range(len(public_dialogues_summary)):
        if i==0: 
            if pursuaded[i] : 
                msgs += user2npc_summary[i] + "그래서 쥐는 빨간 망토와 함께 늑대에게 복수하러 가기로 했다.\n"
                npc_pursuaded += "쥐와 "
            else : msgs += user2npc_summary[i] + "그래서 쥐는 늑대에게 복수하러 가는 빨간 망토와 함께하지 않기로 했다.\n"
        if i==1: 
            if pursuaded[i] : 
                msgs += user2npc_summary[i] + "그래서 삵은 빨간 망토와 함께 늑대에게 복수하러 가기로 했다.\n"
                npc_pursuaded += "삵과 "
            else : msgs += user2npc_summary[i] + "그래서 삵은 늑대에게 복수하러 가는 빨간 망토와 함께하지 않기로 했다.\n"
        else : 
            if pursuaded[i] : 
                msgs += user2npc_summary[i] + "그래서 곰은 빨간 망토와 함께 늑대에게 복수하러 가기로 했다.\n"
                npc_pursuaded += "곰과 "
            else : msgs += user2npc_summary[i] + "그래서 곰은 늑대에게 복수하러 가는 빨간 망토와 함께하지 않기로 했다.\n"
        
        msgs += public_dialogues_summary[i] + "\n"

    if final_game:
        msgs += f"그리고 결국 빨간 망토와 {npc_pursuaded}은(는) 함께 늑대에게 복수하여 물리치는 것을 성공해냈다.\n"
    else : 
        msgs += f"하지만 결국 빨간 망토와 {npc_pursuaded}은(는) 함께 늑대에게 복수하는 데에 실패하고 말았다\n"

    # 프롬프트 구성    
    msgs += "Please make a wonderful fairy tale with the above and more than 30 lines long. Make it detailed in Korean. Write: 'what to say'"

    # ChatGPT API 호출
##    completion = create_gpt_saying([{"role": "user", "content" : msgs}])
##    content = completion["choices"][0]["message"]["content"].strip()
##    story = content
##    result = final_game

    # 전역 변수 초기화
    
    public_dialogues = [""]
    public = [""]
    public_dialogues_summary = []
    messages = [
    {"role": "system", "content": "You're a helpful assistant."},
    {"role": "user", "content": "I want chatgpt to make his own decisions based on the context and make a story."},
    {"role": "assistant", "content": "I will no longer provide specific dialogue guidelines, so please allow ChatGPT to proceed freely. This will allow ChatGPT to leverage the context given to create creative conversations. From now on, I will not limit my response and let ChatGPT continue the conversation freely."},
    ]

    pursuaded = [ False, False, False ] # 쥐, 삵, 곰
    user2npc_summary = ["", "", ""]
    npcs = []
    memories = []
    observations = []
    final_game = True
    story = ""
###    return json.dumps({"content": content, "game_result": result}, ensure_ascii=False)
    return json.dumps({"content": "한때 마을에 살던 빨간 망토는 어느 날, 할머니의 안부를 전하러 가는 도중 늑대가 그녀를 위협하는 걸 목격한다. 분노와 원한으로 가득 찬 망토는 늑대에게 복수하기로 결심한다. 하지만 혼자서 갈 수 없다고 생각해 함께 가야할 동료인 지혜로운 쥐와 함께 계획을 세우기로 한다.\n\n망토와 쥐는 이들의 첫 번째 임무로, 늑대가 숨어있을 것으로 예상되는 집으로 향한다. 이 집은 오래된 나무집이었으며, 그 앞에는 거미줄과 고사리가 얽혀 있었다. 들어갈 때부터 답답함을 느낀 망토와 쥐였지만, 서로를 의지하며 용기를 내어 집 안으로 들어섰다.\n\n집 안은 어두컴컴하여 시야가 제한되었다. 그럼에도 불구하고 망토와 쥐는 조용히 발소리를 줄여 움직이며 방안을 탐색하기 시작한다. 곳곳에는 늑대의 흔적이 남아 있다...", "game_result": True}, ensure_ascii=False)


# NPC

""" 수정 필요
# Dall-E 호출
@application.route('/get_image', methods=['GET'])
def dialoguesImage():
    # 이미지 파일 경로
    image_path = '/image.jpg'

    return send_file(image_path, mimetype='image/jpeg')

"""
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

    n, m = get_agent(name1, 36, traits1, "He is talking with " + name2)
    npcs.append(n); memories.append(m) 
    n, m = get_agent(name2, 36, traits2, "He is talking with " + name1)
    npcs.append(n); memories.append(m) 

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

    observation = run_conversation(
        npcs,
        pin2
    )

    print("이제 시작")
    
    print("content :....")
    
    # npc간 대화 요약
    content = ""
    for p in public:
        content += p

    content += "I want to summarize this conversation in detail within 1000 tokens. Can you summarize it in Korean? Write : 'what to say'"
    print(content)
        
    completion = create_gpt_saying([{"role": "user", "content": content}])
    answer = completion["choices"][0]["message"]["content"].strip()
    public_dialogues_summary.append(answer)
    
    print("public_dialogues_summary")
    print(public_dialogues_summary)
    print("public_dialogues")
    print(public_dialogues)
    print("public")
    print(public)

    return json.dumps({"dialogues": public_dialogues}, ensure_ascii=False) # 수정? public_dialogues + npcs'name / public



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
    return json.dumps({"pursuade":
                           [{"mouse": user2npc_summary[0], "pursuaded_tf": pursuaded[0]},  # response['pursuade'][0]['mouse'] (문자열 값) | response['summary'][0]['pursuaded_tf'] (boolean 값)
                           {"wildcat": user2npc_summary[1], "pursuaded_tf": pursuaded[1]}, # response['pursuade'][1]['wildcat'] (문자열 값) | response['summary'][1]['pursuaded_tf'] (boolean 값)
                           {"bear": user2npc_summary[2], "pursuaded_tf": pursuaded[2]}]    # response['pursuade'][2]['bear'] (문자열 값) | response['summary'][2]['pursuaded_tf'] (boolean 값)
                        }, ensure_ascii=False)


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
    completion = create_gpt_saying([{"role": "user", "content" : msgs}])
    content = completion["choices"][0]["message"]["content"].strip()
    story = content
    result = final_game

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
    return json.dumps({"content": content, "game_result": result}, ensure_ascii=False)


# NPC

""" 수정 필요
# Dall-E 호출
@application.route('/get_image', methods=['GET'])
def dialoguesImage():
    # 이미지 파일 경로
    image_path = '/image.jpg'

    return send_file(image_path, mimetype='image/jpeg')

"""

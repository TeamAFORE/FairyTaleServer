from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS

import math
import faiss # Facebook AI Research에서 개발한 빠른 유사도 검색(Similarity Search) 라이브러리
from termcolor import colored
from typing import List
#from datetime import datetime, timedelta

import logging
logging.basicConfig(level=logging.ERROR)

from experimental import (
    GenerativeAgent,
    GenerativeAgentMemory
)




""" agent들 간의 상호작용 variables """

# 여론 대화 내용  
public_dialogues = [""]
public = [""]

# 여론 대화 내용 요약
public_dialogues_summary = []

# Generative Agent를 인터뷰하는 Player
USER_NAME = "Little red riding hood"

# LLM 설정
LLM = ChatOpenAI(max_tokens=1000)
#LLM1.openai_api_key = "..."
#print(LLM1.openai_api_key)





""" Generative Agent Method """

# 0부터 1까지의 척도로 유사도 반환
def relevance_score_fn(score: float) -> float:
    # 거리, VectorStore에 의해 사용된 유사도 metric, 임베딩 규모(OpenAI는 단위 표준으로, 다른 것들은 그렇지 않다)에 따라 달라진다
    # 정규화된 임베딩의 유클리드 노름을 변환한다 (0 : 가장 유사함, sqrt(2) : 가장 다름)
    return 1.0 - score / math.sqrt(2)


#  특정 에이전트(AI 에이전트)에게만 해당하는 독자적인, 새로운 벡터 저장소 검색기 생성
def create_new_memory_retriever():
    # 임베딩 모델 정의 (임베딩 모델 : 텍스트나 다른 형태의 데이터를 숫자 벡터로 변환하는 기술
    embeddings_model = OpenAIEmbeddings()
    # 벡터 저장소를 비어있는 상태로 초기화
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(
        embeddings_model.embed_query,
        index,
        InMemoryDocstore({}),
        {},
        relevance_score_fn=relevance_score_fn,
    )
    return TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, other_score_keys=["importance"], k=15
    )


# Player의 Generative Agent Pre-Interview
def interview_agent(agent: GenerativeAgent, message: str) -> str:
    new_message = f"{USER_NAME} says {message}"
    return agent.generate_dialogue_response(new_message)[1]
    

# 매 관측시마다 시스템이 생성하는 요약문 확인 및 요약문 발전 감시(for 성능 평가 및 개선)
def summary_character_observation(agent: GenerativeAgent, observations): # observations는 List
    summary_observation = []
    for i, obsv in enumerate(observations):
        _, reaction = agent.generate_reaction(obsv)
        print(colored(obsv, "green"), reaction)
        if ((i+1) % 20) == 0:
            content = f"After {i+1} observations, {agent.name}'s summary is:\n{agent.get_summary(force_refresh=True)}"
            print("*" * 40)
            print(colored(content, "blue"))
            summary_observation.append(content)
            print("*" * 40)


# Generative Agent들 간 대화
def run_conversation(agents: List[GenerativeAgent], initial_observation: str) -> None:
    obsv = []
    obsv.append(initial_observation)
    public_dialogues[0] = str(initial_observation[0])
    print(f"agents : {agents[0].name} and {agents[1].name}")
    public[0] = f"{agents[0].name} said  " + str(initial_observation[0]) + "\n"
    print("처음")
    print(initial_observation)
    _, observation = agents[1].generate_reaction(initial_observation)
    obsv.append(observation)
    public.append(observation + "\n")
    print(observation)
    turns = 0
    for i in range(2): # 대화 횟수 조정
        break_dialogue = False
        for agent in agents:
            stay_in_dialogue, observation = agent.generate_dialogue_response(observation)
            obsv.append(observation)
            public.append(observation + "\n")
            public_dialogues.append(observation[(observation.find('said ')+6):(len(observation)-1)])
            print(observation)
            # observation = f"{agent.name} said {reaction}
            if not stay_in_dialogue:
                break_dialogue = True
        if break_dialogue:
            break
        turns += 1
    return obsv



""" 사용 예시 

# Generative Agent의 Memory 생성
tommies_memory = GenerativeAgentMemory(
    llm=LLM,
    memory_retriever=create_new_memory_retriever(),
    verbose=False,
    reflection_threshold=8,  # we will give this a relatively low number to show how reflection works
)

eves_memory = GenerativeAgentMemory(
    llm=LLM,
    memory_retriever=create_new_memory_retriever(),
    verbose=False,
    reflection_threshold=5,
)

# Generative Agent 생성
tommie = GenerativeAgent(
    name="Tommie",
    age=25,
    traits="anxious, likes design, talkative",  # NPC의영구적인 특성
    status="looking for a job",  # Virtual world에 연결 시, status 업데이트 가능
    memory_retriever=create_new_memory_retriever(),
    llm=LLM,
    memory=tommies_memory,
)

eve = GenerativeAgent(
    name="Eve",
    age=34,
    traits="curious, helpful",  # 영구적인 특성 추가 가능
    status="N/A",  # Virtual world와 연결될 때, 업데이트 가능
    llm=LLM,
    daily_summaries=[
        (
            "Eve started her new job as a career counselor last week and received her first assignment, a client named Tommie."
        )
    ],
    memory=eves_memory,
    verbose=False,
)

# memory 객체에 직접적으로 기억(혹은 관측)들을 추가 가능
tommie_observations = [
    #"Tommie remembers his dog, Bruno, from when he was a kid",  # 어릴 적 브루노라는 개를 기억한다
    "토미는 먼 거리를 운전하느라 피곤하다" # "Tommie feels tired from driving so far", # 먼 거리를 운전하느라 피곤하다
    #"Tommie sees the new home", # 새로운 집을 보다
    "새로운 이웃들은 고양이를 키운다" #"The new neighbors have a cat", # 새로운 이웃들은 고양이를 키운다
    #"The road is noisy at night", # 밤에는 길이 시끄러워 소음이 있다
    #"Tommie is hungry", # 배고프다
    #"토미는 휴식을 취한다" #"Tommie tries to get some rest.", # 휴식을 취한다
]

eve_observations = [ # 하루 동안 겪은 일(관측 등)
    #"Eve wakes up and hear's the alarm",
    #"이브는 죽 한 그릇을 먹는다." # "Eve eats a boal of porridge",
    #"Eve helps a coworker on a task",
    "이브는 그녀의 친구 Xu와 일을 가기 전에 테니스를 친다." # "Eve plays tennis with her friend Xu before going to work",
    "이브는 그녀의 동료가 토미와 함께 일하는 것은 힘들다고 말하는 것을 듣게 되었다" # "Eve overhears her colleague say something about Tommie being hard to work with",
]

print("토미 관찰")
for observation in tommie_observations:
    tommie.memory.add_memory(observation)

print("이브 관찰")
for observation in eve_observations: # memory에 관측 추가
    eve.memory.add_memory(observation)

# Generative Agent 기억 요약 및 재구성
tommie.get_summary(force_refresh=True)

# Generative Agent들 간의 대화
agents = [tommie, eve]
run_conversation(
    agents,
    "Tommie said: Hi, Eve. Thanks for agreeing to meet with me today. I have a bunch of questions and am not sure where to start. Maybe you could first share about your experience?",
)

# Player의 Generative Agent 인터뷰
interview_agent(tommie, "How was your conversation with Eve?")

# memory에 관측 추가
yesterday = (datetime.now() - timedelta(days=1)).strftime("%A %B %d") # 어제 날짜
eve_observations = [ # 하루 동안 겪은 일(관측 등)
    "Eve wakes up and hear's the alarm",
    "Eve eats a boal of porridge",
    "Eve helps a coworker on a task",
    "Eve plays tennis with her friend Xu before going to work",
    "Eve overhears her colleague say something about Tommie being hard to work with",
]
for observation in eve_observations: # memory에 관측 추가
    eve.memory.add_memory(observation)
"""

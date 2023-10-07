import openai
from character_interaction import *
from game_setting import *

""" AWS SECRET MANAGER """
# Use this code snippet in your app.
# If you need more information about configurations or implementing the sample code, visit the AWS docs:
# https://aws.amazon.com/developers/getting-started/python/

import boto3 # Python을 AWS CLI에서 사용하기 위한 AWS SDK(Software Development Kit)
import base64
from botocore.exceptions import ClientError

def get_secret():

    secret_name = "fairy_tale/openai/api_key"
    region_name = "ap-northeast-2"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        if e.response['Error']['Code'] == 'DecryptionFailureException':
            # Secrets Manager can't decrypt the protected secret text using the provided KMS key.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InternalServiceErrorException':
            # An error occurred on the server side.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidParameterException':
            # You provided an invalid value for a parameter.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidRequestException':
            # You provided a parameter value that is not valid for the current state of the resource.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'ResourceNotFoundException':
            # We can't find the resource that you asked for.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e

    else:
        # Decrypts secret using the associated KMS key.
        # Depending on whether the secret is a string or binary, one of these fields will be populated.
        if 'SecretString' in get_secret_value_response:
            return get_secret_value_response['SecretString']
        else:
            return base64.b64decode(get_secret_value_response['SecretBinary'])

    # Decrypts secret using the associated KMS key.
    secret = get_secret_value_response['SecretString']




""" system variables """

""" user2npc """
# npc 설득 세팅
messages = [
    {"role": "system", "content": "You're a helpful assistant."},
    {"role": "user", "content": "I want chatgpt to make his own decisions based on the context and make a story."},
    {"role": "assistant", "content": "I will no longer provide specific dialogue guidelines, so please allow ChatGPT to proceed freely. This will allow ChatGPT to leverage the context given to create creative conversations. From now on, I will not limit my response and let ChatGPT continue the conversation freely."},
    ]

# npc 설득 여부
pursuaded = [ False, False, False ] # 쥐, 삵, 곰

# npc 설득 요약
user2npc_summary = ["", "", ""]


""" npc2npc """
npcs = []
memories = []
observations = []

""" final game """
final_game = True

""" 최종 story """
story = ""




""" Character-Character API methods """

# agent(character), memory 생성
def get_agent(name, age, traits, status):  
    LLM = ChatOpenAI(max_tokens=1000)


    memory = GenerativeAgentMemory(
        llm=LLM,
        memory_retriever=create_new_memory_retriever(),
        verbose=False,
        reflection_threshold=8,
    )

    agent = GenerativeAgent(
        name=name,
        age=age,
        traits=traits,  # 성격 : 차분한, 활발한, 수다스러운
        status=status,  # 상황 : 일자리를 구하고 있다,
        memory_retriever=create_new_memory_retriever(),
        llm=LLM,
        memory=memory
    )

    return agent, memory



""" User-Character API methods """

# Character 생성
def start_to_talking(npc_name):
    messages.append({"role": "user", "content": f"Act as a {npc_name} in a red cape fairy tale. I'll act as the main character, the red cape. An anger index can be expressed as a percentage. The most angry state is 100%, and the least angry state is 0%. Whenever You think wolves are really bad during this conversation with me, your anger level will increase by 20%. We'll take turns talking. Please speak in dialogue within 4 sentences. I don't need any explanation. Just talk to me."})
    messages.append({"role": "assistant", "content": "Sure, I'm ready."})

    if npc_name == "mouse":
        messages.append({"role": "user", "content" : user2npc_story[0]})
    elif npc_name == "wildcat":
        messages.append({"role": "user", "content" : user2npc_story[1]})
    elif npc_name == "bear" :
        messages.append({"role": "user", "content" : user2npc_story[2]})

    messages.append({"role": "assistant", "content": f"Right. so we turn into red capes and {npc_name}."})
    messages.append({"role": "user", "content": "But you have no intention of following me to avenge the wolf. Never. Your anger towards the wolf is about 10%, and my anger towards the wolf is about 80%. You can follow me after hearing my persuasion to avenge the wolf. And you might not follow me. After about 5 conversations, you'll decide for yourself. Must be creative in dialogue. You'll go first. Say all the answers in Korean."})
    messages.append({"role": "assistant", "content": "빨간 망토야, 너의 할머니가 늑대에게 잡아먹힐 뻔 했다는 얘기를 들었어. 정말 화가 많이 났겠다."})


# ChatGPT API 호출
def create_gpt_saying(dialogue):
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=dialogue,
        max_tokens=400,  # 생성할 최대 토큰 수
        temperature=0.9,  # 출력 다양성 조절 0.2 ~ 1.0
        n=1,  # 생성할 응답 수
        stop="exit",  # 생성을 멈출 단어 또는 문장의 목록. None일 경우 토큰 소요시까지
        top_p=0.8,  # 생성에 사용할 토큰의 확률 사이의 경계. 이보다 높은 확률을 가진 토큰만 고려된다
        frequency_penalty=0.7,  # 응답에서 일반적인 단어를 제외하는 데 사용되는 매개변수. 높은 값은 일반적인 단어를 제거하고, 낮은 값은 더 일반적인 단어를 포함한다.
        presence_penalty=0.9,  # 응답에서 반복되는 단어를 줄이는 데 사용된다
        timeout=None  # API 응답이 돌아올 때 설정한 시간을 넘으면 오류를 띄움
    )

# Character 초기화
def _init_prompt():
    global messages
    messages = messages[:3]
    return messages

# 대화
def talk_with_npc(npc_name, user_speaking):
    # User 대화 : user_speaking = input("user: ")

    # NPC name
    name = npc_name

    # 대화 마칠 때 == pursuaded_tf & summary & init
    if user_speaking == "exit":
        print("대화를 마쳤습니다.")

        # 설득 여부 message에 저장
        messages.append({"role": "user", "content": "좋아! 그러면 네가 나와 함께 복수하러 가는 게 맞다면 write:\nTrue, 아니라면 write:\nFalse"})
        completion = create_gpt_saying(messages)

        assistant_content = completion["choices"][0]["message"]["content"].strip()
        if name == "mouse":
            if assistant_content == "True": pursuaded[0] = True
            else: pursuaded[0] = False
        elif name == "wildcat":
            if assistant_content == "True": pursuaded[1] = True
            else: pursuaded[1] = False
        elif name == "bear":
            if assistant_content == "True": pursuaded[2] = True
            else: pursuaded[2] = False

        messages.append({"role": "assistant", "content": f"{assistant_content}"})
        completion['messages'] = messages

        # 요약
        messages.append({"role": "user", "content": '이제 지금까지의 대화를 요약해서 하나의 이야기를 만들어줘 write: "what to say"'})
        summary = create_gpt_saying(messages)
        if name == "mouse": user2npc_summary[0] = summary["choices"][0]["message"]["content"].strip()
        elif name == "wildcat": user2npc_summary[1] = summary["choices"][0]["message"]["content"].strip()
        elif name == "bear": user2npc_summary[2] = summary["choices"][0]["message"]["content"].strip()

        # messages(npc 프롬프트) 초기화
        _init_prompt()
        print(pursuaded)

        return convert_to_dic(completion)



    # User 응답 저장
    messages.append({"role": "user", "content": f"{user_speaking}"})

    # ChatGPT API 호출
    completion = create_gpt_saying(messages)
    # ChatGPT 응답 저장
    assistant_content = completion["choices"][0]["message"]["content"].strip()
    messages.append({"role": "assistant", "content": f"{assistant_content}"})
    completion['messages'] = messages

    # --서버 쪽 확인용 코드
    print("user: ", user_speaking)
    print("npc: ", assistant_content)

    # ChatGPT 응답 JSON 데이터 -> String으로 변환하여 반환
    return convert_to_dic(completion)
    

# ChatGPT 응답 객체(OpenAIObject)를 JSON으로 변환
def convert_to_dic(completion):
    dic_data = {
        # 'id': completion["id"],
        # 토큰 수 (completion_tokens, prompt_tokens, total_tokens)
        'prompt_tokens': completion["usage"]["prompt_tokens"],
        # 'completion_tokens': completion["usage"]["completion_tokens"],
        'total_tokens': completion["usage"]["total_tokens"],
        # 응답자(assistant)
        'role': completion["choices"][0]["message"]["role"],
        'content': completion["choices"][0]["message"]["content"],
        'messages': completion["messages"]
    }

    return dic_data


def summarize_dialogue(character): # --> 현재 오류 : 한 번 시도해보고 안 되면 말 것 -> 현영
    messages.append({"role": "system",
                     "content": f"You're a helpful assistant. Stop acting as a {character} and come back to ChatGPT."})
    messages.append({"role": "assistant", "content": "Okay, now I'm back on ChatGPT. What can I do for you?"})
    messages.append({"role": "user",
                     "content": f"Please summarize the conversation between {character} and red cape so far and turn it into a story. Say all the answers in Korean."})
    completion = create_gpt_saying(messages)
    assistant_content = completion["choices"][0]["message"]["content"].strip()

    return assistant_content



def pickUpKeywords(character1, data=messages, character2="red cape"):  # messages, bear, red cape

    code = [  # 키워드 추출 요청 프롬프트
        {"role": "system", "content": "You're a helpful assistant."},
        {"role": "user", "content": f"Stop acting as a {character1} and come back to ChatGPT."},
        {"role": "assistant", "content": "Okay, now I'm back on ChatGPT. What can I do for you?"},
        {"role": "user",
         "content": f"I'll give you the list data. This is a collection of conversations between {character1} and {character2}"},
        {"role": "assistant",
         "content": f"Great! I'll help you learn the model based on the conversation data of {character1} and {character2} to respond appropriately. You can provide the data in a format where I can learn the model."},

    ]

    code.append(
        {"role": "user", "content": f"{data} 이 중에서 한글 키워드 5개 추출해줘. 각 키워드를 List 형태로 반환해줘. 그 어떤 설명도 필요없어. 키워드만 줘."})
    completion = create_gpt_saying(code)
    assistant_content = completion["choices"][0]["message"]["content"]
    # -- 서버 쪽 확인용 코드
    # print(assistant_content)

    # code.append({"role": "assistant", "content": f"{assistant_content}"})
    # print(f"GPT: {assistant_content}")

    print(assistant_content.split("/"))
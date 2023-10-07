# !pip install langchain-experimental

import re # 정규 표현식을 사용하기 위한 모듈
from datetime import datetime # 날짜와 시간을 다루기 위한 모듈
from typing import Any, Dict, List, Optional, Tuple # 타입 힌트를 위한 모듈. 다양한 데이터 유형

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate # 대화 흐름의 템플릿화된 프롬프트를 생성하기 위해 사용
from langchain.schema.language_model import BaseLanguageModel # 언어 모델의 기본 클래스로, 다양한 언어 모델의 공통 인터페이스를 정의
from pydantic import BaseModel, Field # pydantic. 데이터 모델 유효성 검사 및 직렬화를 위한 도구로 사용되는 패키지 > 데이터 모델 지정 및 구성 옵션 지정

# langchain_experimental 패키지 사용하는 대신 직접 수정한 코드를 사용해야 하는 경우 주석 풀고 아래와 변경하기
#import import_ipynb # ipynb 파일을 가져오기 위한 모듈. (jupyter notebook)
from experimental.memory import GenerativeAgentMemory # 생성 에이전트의 기억을 관리하는 클래스로, 관찰 및 메모리 추가와 같은 작업을 수행
"""
from langchain_experimental.generative_agents.memory import GenerativeAgentMemory # 생성 에이전트의 기억을 관리하는 클래스로, 관찰 및 메모리 추가와 같은 작업을 수행
"""

"""
GenerativeAgent 클래스 
: NPC의 속성과 행동을 모델링하기 위한 데이터 구조를 제공하는 클래스.
: 이 객체를 사용하여 NPC의 정보 저장, 기억 추가, 요약 정보 생성, 다양한 상호작용 구현
: 즉, 우리 게임에서 GenerativeAgent를 NPC라고 볼 수도 있음
"""


class GenerativeAgent(BaseModel):  # BaseModel : 사용하는 데이터 모델
    """GenerativeAgent = '기억'과 '개별적 서사 또는 성격'을 가진 'NPC'"""

    # NPC 기본 정보
    name: str
    age: Optional[int] = None  # Optional : typing 모듈에서 제공되는 제네릭(Generic) 타입. Optional[int]는 int형이거나 None이거나.
    traits: str = "N/A"  # NPC에게 부여된 영구적인 특성을 경나타내는 문자열.
    status: str  # NPC의 특성 중 우리가 생각하기에 변화되지 않기를 원하는 특성을 나타내는 문자열
    memory: GenerativeAgentMemory  # NPC의 기억 객체. 기억은 관련성(relevance), 최신성(recency) 및 '중요도(importance)'를 결합

    llm: BaseLanguageModel  # 기본 언어 모델

    verbose: bool = False  # 상세한, 더 많은 정보를 포함할 메시지를 출력할지 여부

    summary: str = ""  #: :meta private:  # NPC의 기억을 바탕으로 reflection하며 자기 자신에 대한 간단한 상태 정보를 기억에 생성하고 유지함을 나타내는 문자열
    summary_refresh_seconds: int = 3600  #: :meta private: # NPC 요약 정보를 얼마나 자주 다시 생성할지를 나타내는 정수값. 3600sec = 1h
    last_refreshed: datetime = Field(
        default_factory=datetime.now)  # : :meta private: # NPC 요약 정보가 마지막으로 생성되어 갱신된 시간. 기본값 = 현재시간
    daily_summaries: List[str] = Field(
        default_factory=list)  # : :meta private: # NPC가 수행한 계획의 이벤트에 대한 요약 정보를 담고 있는 문자열의 리스트

    # @고민정 : 게임 진행을 위한 추가 속성
    user_used : bool = False



    class Config:  # Pydantic 객체의 동작 및 설정을 사용자 지정하기 위해 제공되는 내부 클래스
        """
        Pydantic : 데이터 유효성 검사와 직렬화를 위한 Python 데이터 모델링 라이브러리
        데이터를 정의하고 유효성을 검사하는 간단하고 명확한 방법으로 데이터 모델 작성 가능
        """

        arbitrary_types_allowed = True  # True : Pydantic 객체에 임의의 유형 사용 가능
        # 즉, Pydantic이 사전 정의된 유효성 검사 규칙 외에도 사용자 지정 유효성 검사 규칙을 수용할 수 있도록 함
        # 단, 임의의 유형을 사용할 경우 Pydantic의 데이터 유효성 검사와 자동 직렬화/역직렬화 기능을 활용할 수 없으며, 사용자가 직접 유효성 검사 및 직렬화/역직렬화 로직을 구현해야 함

    # @고민정
    def printAgent(self):
        llm = self.memory.llm
        memory_retriever = self.memory.memory_retriever

        print(f"{self.name} : " + "llm : " + str(llm) + " 메모리리트라이버 : " + str(memory_retriever))
        return memory_retriever

    # LLM-related methods
    @staticmethod
    def _parse_list(text: str) -> List[str]:
        lines = re.split(r"\n", text.strip())  # 주어진 문자열 양쪽 공백 제거 -> 개행 문자 기준 문자열 반환 및 리스트로 반환
        return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in
                lines]  # ^(문자열 시작)에서 0개 이상 공백 문자, 1개 이상 숫자, 온점, 0개 이상 공백문자를 ""(빈 문자열)로 대체

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(  # LLMChain : 언어 생성 모델(Language Model)과 상호작용하기 위한 인터페이스
            llm=self.llm, prompt=prompt, verbose=self.verbose, memory=self.memory  # verbose : 출력 로그를 제어
        )

    # 주어진 관측(observation)에서 관찰 대상 엔티티(entity)를 추출
    def _get_entity_from_observation(self, observation: str) -> str:
        prompt = PromptTemplate.from_template(
            "What is the observed entity in the following observation? {observation}"  # 다음 관측에서 관찰된 객체는 무엇인가? {observation} 변수 ex) The dog is eating a meat.
            + "\nEntity="
        )
        return self.chain(prompt).run(observation=observation).strip()

    # 주어진 관측(observation)과 관찰 대상 엔티티 이름 => 해당 엔티티 동작(action) 추출
    def _get_entity_action(self, observation: str, entity_name: str) -> str:
        prompt = PromptTemplate.from_template(
            "What is the {entity} doing in the following observation? {observation}"  # 주어진 관측에서 {entity}는 무엇을 하고 있는가? {observation}
            + "\nThe {entity} is"
        )
        return (
            self.chain(prompt).run(entity=entity_name, observation=observation).strip()
        )

    # 관측과 가장 연관된 기억(memory)를 요약
    def summarize_related_memories(self, observation: str) -> str:
        prompt = PromptTemplate.from_template(
            # relevant_memories : 추후 채워질 기억들
            """
{q1}?
Context from memory:
{relevant_memories}
Relevant context: 
"""
        )
        entity_name = self._get_entity_from_observation(observation)  # 현재 NPC와 대화하는 NPC entity 이름
        entity_action = self._get_entity_action(observation, entity_name)  # 현재 NPC와 대화하는 NPC entity 행동
        q1 = f"What is the relationship between {self.name} and {entity_name}"  # 현재 NPC와 대화하는 NPC entity와의 관계는 무엇인가?
        q2 = f"{entity_name} is {entity_action}"  # 상대 NPC는 ~'action'하고 있다
        return self.chain(prompt=prompt).run(q1=q1, queries=[q1, q2]).strip()

    # 관측 및 대화에 대한 반응 생성
    def _generate_reaction(
            self, observation: str, suffix: str, now: Optional[datetime] = None  # now : 현재 또는 now 시간
    ) -> str:

        prompt = PromptTemplate.from_template(
            "{agent_summary_description}"
            + "\nIt is {current_time}."  # 지금은 {current_time}이다
            + "\n{agent_name}'s status: {agent_status}"  # {agent_name}의 상태: {agent_status}
            + "\nSummary of relevant context from {agent_name}'s memory:"  # {agent_name}의 기억 속 관련 컨텍스트 요약
            + "\n{relevant_memories}"
            + "\nMost recent observations: {most_recent_memories}"  # 최근의 관측: {most_recent_memories}
            + "\nObservation: {observation}"  # 관측: {observation}
            + "\n\n"
            + suffix
        )
        agent_summary_description = self.get_summary(now=now)  # 시간 now일 때의 NPC 성격, 특성 등의 요약
        relevant_memories_str = self.summarize_related_memories(observation)  # 관측과 연관된 기억 요약
        current_time_str = (
            datetime.now().strftime("%B %d, %Y, %I:%M %p")  # 현재 월, 일, 연도, 12시간 형식 시간, 분, 오전/오후
            if now is None
            else now.strftime("%B %d, %Y, %I:%M %p")  # 주어진 시간 월, 일, 연도, 12시간 형식 시간, 분, 오전/오후
        )
        kwargs: Dict[str, Any] = dict(
            agent_summary_description=agent_summary_description,
            current_time=current_time_str,
            relevant_memories=relevant_memories_str,
            agent_name=self.name,
            observation=observation,
            agent_status=self.status,
        )
        consumed_tokens = self.llm.get_num_tokens(  # 프롬프트 토큰 수
            prompt.format(most_recent_memories="", **kwargs)
        )
        kwargs[self.memory.most_recent_memories_token_key] = consumed_tokens  # 특정 기억 토큰의 위치를 추적하고 기록
        return self.chain(prompt=prompt).run(**kwargs).strip()

    # 텍스트 특정 패턴 제거
    def _clean_response(self, text: str) -> str:
        return re.sub(f"^{self.name} ", "", text.strip()).strip()

    # 주어진 관측에 대한 반응 생성
    def generate_reaction(
            self, observation: str, now: Optional[datetime] = None
    ) -> Tuple[bool, str]:

        call_to_action_template = (
                "Should {agent_name} react to the observation, and if so,"  # NPC가 반응해야 하는가?
                + " what would be an appropriate reaction? Respond in one line."  # 그렇다면 어떤 것이 적절한 반응인가? 한 줄로 대답하라
                + ' It should only respond through conversation, write:\nSAY: "what to say"'  # 반응으로써 대화하는 경우 SAY : ~ 형식으로 답하라
                # @고민정 : 게임에서는 대화만 필요.
                #+ "\notherwise, write:\nREACT: reaction (if anything)."  # 대화에 참여하지 않고 반응만 하는 경우 REACT : ~ 형식으로 답하라
                #+ "\nEither react, or say something but not both.\n\n"  #  반응 / 대화 중 하나만 선택 가능
        )
        full_result = self._generate_reaction(  # NPC 반응 생성
            observation, call_to_action_template, now=now
        )
        result = full_result.strip().split("\n")[0]
        # AAA
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} observed "  # 관찰/반응 메모리에 저장
                                            f"{observation} and reacted by {result}",
                self.memory.now_key: now,
            },
        )
        if "REACT:" in result:  # NPC가 반응, but 대화에 참여하지 않음
            reaction = self._clean_response(result.split("REACT:")[-1])
            return False, f"{self.name} (({reaction}))"
        if "SAY:" in result:  # NPC가 대화에 참여
            said_value = self._clean_response(result.split("SAY:")[-1])
            return True, f"{self.name} said {said_value}"
        #else:
        #    return False, result  # NPC가 아무런 반응하지 않음

    # 주어진 관측에 대한 대화 생성
    def generate_dialogue_response(
            self, observation: str, now: Optional[datetime] = None
    ) -> Tuple[bool, str]:

        call_to_action_template = (
            "What would {agent_name} say? To end the conversation, write:"
            ' GOODBYE: "what to say". Otherwise to continue the conversation,'
            ' write: SAY: "what to say next"\n\n'
        )
        full_result = self._generate_reaction(
            observation, call_to_action_template, now=now
        )
        result = full_result.strip().split("\n")[0]
        if "GOODBYE:" in result:
            farewell = self._clean_response(result.split("GOODBYE:")[-1])
            self.memory.save_context(
                {},
                {
                    self.memory.add_memory_key: f"{self.name} observed "
                                                f"{observation} and said {farewell}",
                    self.memory.now_key: now,
                },
            )
            return False, f"{self.name} said {farewell}"
        if "SAY:" in result:
            response_text = self._clean_response(result.split("SAY:")[-1])
            self.memory.save_context(
                {},
                {
                    self.memory.add_memory_key: f"{self.name} observed "
                                                f"{observation} and said {response_text}",
                    self.memory.now_key: now,
                },
            )
            return True, f"{self.name} said {response_text}"
        else:
            return False, result

            ################################################################################################

    # Agent stateful' summary methods.                                                             #
    # Each dialog or response prompt includes a header summarizing the agent's self-description.   #
    # This is updated periodically through probing its memories                                    #
    ################################################################################################

    # NPC의 핵심 특징 요약 : 핵심적인 내용만 짧게
    def _compute_agent_summary(self) -> str:

        prompt = PromptTemplate.from_template(
            "How would you summarize {name}'s core characteristics given the"  # 다음 문장들을 고려하여 NPC{name}의 핵심 특징을 어떻게 요약할 수 있을까요?
            + " following statements:\n"
            + "{relevant_memories}"
            + "Do not embellish."  # 가능한 한 간결하게 작성해주세요
            + "\n\nSummary: "
        )
        # The agent seeks to think about their core characteristics.
        return (
            self.chain(prompt)
            .run(name=self.name, queries=[f"{self.name}'s core characteristics"])
            .strip()
        )

    # Agent에 대한 요약을 반환 및 새로고침 여부 : Agent에 대한 서술적인 요약을 반환 / Agent의 이름, 나이, 내재적 특성 및 요약 결과를 포함하는 문자열을 반환
    def get_summary(
            self, force_refresh: bool = False, now: Optional[datetime] = None
    ) -> str:

        current_time = datetime.now() if now is None else now
        since_refresh = (current_time - self.last_refreshed).seconds
        if (
                not self.summary
                or since_refresh >= self.summary_refresh_seconds
                or force_refresh  # force_refresh : 요약을 강제로 새로고침할지 여부
        ):
            self.summary = self._compute_agent_summary()
            self.last_refreshed = current_time
        age = self.age if self.age is not None else "N/A"
        return (
                f"Name: {self.name} (age: {age})"
                + f"\nInnate traits: {self.traits}"
                + f"\n{self.summary}"
        )

    # Agent의 상태, 요약 및 현재 시간을 포함하는 전체 헤더를 반환하는 기능
    def get_full_header(
            self, force_refresh: bool = False, now: Optional[datetime] = None
    ) -> str:

        now = datetime.now() if now is None else now
        summary = self.get_summary(force_refresh=force_refresh, now=now)
        current_time_str = now.strftime("%B %d, %Y, %I:%M %p")
        return (
            f"{summary}\nIt is {current_time_str}.\n{self.name}'s status: {self.status}"
        )








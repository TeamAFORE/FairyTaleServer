import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import TimeWeightedVectorStoreRetriever
#from experimental.custom_time_weighted_retriever import TimeWeightedVectorStoreRetriever # TF-IDF 벡터화 및 코사인 유사도 활용으로 관련도 높은 데이터 조회하기 위한 클래스 사용
from langchain.schema import BaseMemory, Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.utils import mock_now



logger = logging.getLogger(__name__)


class GenerativeAgentMemory(BaseMemory):
    llm: BaseLanguageModel  # 핵심 언어 모델

    memory_retriever: TimeWeightedVectorStoreRetriever  # 현재 상황과 관련된 기억을 검색

    verbose: bool = False  # 출력되는 정보의 양을 다양하게 하고, 상세하게 할지에 대한 여부

    reflection_threshold: Optional[
        float] = None  # aggregate_importance가 임계값 reflection_threshold을 초과하면 반영(relection)을 멈춘다
    # aggregate_importance = 최근 기억들의 중요성(importance)을 합산한 값

    current_plan: List[str] = []  # NPC의 현재 계획

    # A weight of 0.15 makes this less important than it
    # would be otherwise, relative to salience and time
    importance_weight: float = 0.15  # 기억의 중요성에 가중치를 얼마나 할당할지. 그렇지 않은 경우보다 0.15만큼 덜 중요하게 여긴다
    # 중요성은 현저성(salience), 시간(time)과 비교하여 평가된다

    aggregate_importance: float = 0.0  # : :meta private: # 최근 기억들의 중요성(importance)를 합산하여 추적. reflection_threshold 임계값에 도달하면 반영(reflection)을 멈춘다    """Track the sum of the 'importance' of recent memories.

    max_tokens_limit: int = 800  # : :meta private: # 최대 토큰

    # input keys # 입력 데이터의 구분을 위해 사용
    queries_key: str = "queries"  # 질의(query)는 "queries"라는 키를 사용하여 전달된다
    most_recent_memories_token_key: str = "recent_memories_token"  # 최근 기억의 토큰 데이터에 대한
    add_memory_key: str = "add_memory"  # 새로운 기억을 추가 및 저장하거나 관리할 때 이 키를 사용하여 데이터를 구

    # output keys # 출력 데이터의 구분을 위해 사
    relevant_memories_key: str = "relevant_memories"  # 관련 기억들을 저장하거나 반환할 때 이 키를 사용하여 데이터를 식별
    relevant_memories_simple_key: str = "relevant_memories_simple"  # 간단한 형태의 관련 기억 데이터에 대한 키. 간략화된 형태의 관련 기억 데이터를 저장하거나 반환할 때 사용
    most_recent_memories_key: str = "most_recent_memories"  # 최근 기억을 저장하거나 반환할 때 이 키를 사용하여 데이터를 구분
    now_key: str = "now"  # 현재 시간 데이터에 대한 키. 현재 시간 관련 정보 저장하거나 반환할 때 이 키를 사용하여 데이터를 구

    reflecting: bool = False  # 반영 : 기억을 재생성하고 상황에 맞게 활용하는 것. True : 시스템이 반영 상태이다. False : 시스템은 반영 상태가 아니므로 다른 기억을 고려하지 않을 수도 있다

    # @고민정 : 게임 진행을 위한 추가 속성
    user_used: bool = False



    # LLMChain 클래스 객체 생성
    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)

    # 주어진 텍스트의 '\n' 개행문자를 기준으로 구분하여 리스트에 담는다
    @staticmethod
    def _parse_list(text: str) -> List[str]:

        lines = re.split(r"\n", text.strip())  # text를 \n을 기준으로 분할하여 lines 리스트에 저장하고 text의 앞 뒤 공백을 제거
        lines = [line for line in lines if line.strip()]  # lines 리스트에서 공백만 있는 빈 줄을 제거
        return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in
                lines]  # 문자열 시작(^)부터 숫자(d+)+점(.)+공백(s*) 패턴을 빈 문자열로 대체함으로써 제거한다. strip() : line 문자열의 앞 뒤 공백을 최종적으로 제거

    # 최근 기억에 대해 가장 현저한(주목할만 한, 두드러진) 3개의 고수준 질문을 생성하여 반환
    def _get_topics_of_reflection(self, last_k: int = 10) -> List[str]:

        prompt = PromptTemplate.from_template(
            "{observations}\n\n"  # 관측한 것들
            "Given only the information above, what are the 3 most salient "  # 오직 위의 정보만이 주어졌을 때 가장 중요한(주목할만한) 
            "high-level questions we can answer about the subjects in the statements?\n"  # 고수준의 질문 3개는 무엇인가? 단 주어진 정보 안에서 그 주제에 우리가 답할 수 있는 질문이어야 한다
            "Provide each question on a new line."  # 각 질문은 개행문자(\n)로 구분하여 나타낸디
        )
        observations = self.memory_retriever.memory_stream[-last_k:]  # 관측은 기억_조회.기억_스트림 함수를 통해 가져온다
        observation_str = "\n".join(  # 리스트 내의 문자열을 줄 바꿈으로 구분하여 하나의 문자열로 합친다
            [self._format_memory_detail(o) for o in observations]
        )
        result = self.chain(prompt).run(observations=observation_str)  # llm 객체를 생성하고 위의 내용을 prompt에 넣는다
        return self._parse_list(result)

    # 관련된 중요한 기억들에 기반하여 특정 주제(a topic of relfection)에 대한 통찰(insights)을 생성
    def _get_insights_on_topic(
            self, topic: str, now: Optional[datetime] = None
    ) -> List[str]:

        prompt = PromptTemplate.from_template(
            "Statements relevant to: '{topic}'\n"  # topic과 관련된 기억들(Statements)
            "---\n"
            "{related_statements}\n"
            "---\n"
            "What 5 high-level novel insights can you infer from the above statements "  # 주어진 기억들을 기반으로 얻을 수 있는 5가지 높은 수준의 새로운 통찰력들에 대한 질문. 통찰력은 주제에 대한 질문에 관련성이 있어야 한다
            "that are relevant for answering the following question?\n"
            "Do not include any insights that are not relevant to the question.\n"  # 질문과 관련없는 통찰력은 포함하지 말 것
            "Do not repeat any insights that have already been made.\n\n"  # 이미 언급된 통찰력을 중복하여 포함하지 말 
            "Question: {topic}\n\n"
            "(example format: insight (because of 1, 5, 3))\n"  # 통챨력이 어떤 기억들에 기반하여 도출되었는지 번호 리스트 반
        )

        related_memories = self.fetch_memories(topic, now=now)
        related_statements = "\n".join(
            [
                self._format_memory_detail(memory, prefix=f"{i + 1}. ")
                for i, memory in enumerate(related_memories)
            ]
        )
        result = self.chain(prompt).run(
            topic=topic, related_statements=related_statements
        )
        # TODO: Parse the connections between memories and insights
        return self._parse_list(result)

    # 최근 관측에 대해 반영하고, 각 주제에 대한 통찰력 생성하여 리스트로 반환
    def pause_to_reflect(self, now: Optional[datetime] = None) -> List[str]:

        if self.verbose:  # 디버그 정보
            logger.info("Character is reflecting")
        new_insights = []
        topics = self._get_topics_of_reflection()
        for topic in topics:
            insights = self._get_insights_on_topic(topic, now=now)
            for insight in insights:
                self.add_memory(insight, now=now)
            new_insights.extend(insights)
        return new_insights

    # 하나의 주어진 기억(memory)의 절대적 중요을 점수로 평가
    def _score_memory_importance(self, memory_content: str) -> float:

        prompt = PromptTemplate.from_template(
            "On the scale of 1 to 10, where 1 is purely mundane"  # 1을 평범하고 일상적인 것(예를 들어, 이 닦기, 잘 준비하기)으로 했을 때 1~10의 척도
            + " (e.g., brushing teeth, making bed) and 10 is"  # 10은 매우 강한 감정을 일으키는 것(관계가 끝남, 대학 입학 등)
            + " extremely poignant (e.g., a break up, college"
            + " acceptance), rate the likely poignancy of the"  # 특정 기억의 감동적인 정도를 점수로 평가.
            + " following piece of memory. Respond with a single integer."  # 단일 정수로 응답
            + "\nMemory: {memory_content}"
            + "\nRating: "
        )
        score = self.chain(prompt).run(memory_content=memory_content).strip()
        if self.verbose:
            logger.info(f"Importance score: {score}")
        match = re.search(r"^\D*(\d+)", score)
        if match:
            return (float(match.group(1)) / 10) * self.importance_weight
        else:
            return 0.0

    # 여러 개의 주어진 기억들(memories)의 절대적 중요을 점수로 평가 -> 각 기억에 대해 중요성 점수들의 리스트를 반
    def _score_memories_importance(self, memory_content: str) -> List[float]:

        prompt = PromptTemplate.from_template(
            "On the scale of 1 to 10, where 1 is purely mundane"  # 1을 평범하고 일상적인 것(예를 들어, 이 닦기, 잘 준비하기)으로 했을 때 1~10의 척도
            + " (e.g., brushing teeth, making bed) and 10 is"  # 10은 매우 강한 감정을 일으키는 것(관계가 끝남, 대학 입학 등)
            + " extremely poignant (e.g., a break up, college"
            + " acceptance), rate the likely poignancy of the"  # 특정 기억의 감동적인 정도를 점수로 평가.
            + " following piece of memory. Always answer with only a list of numbers."  # 항상 숫자 리스트로 답하라
            + " If just given one memory still respond in a list."  # 하나의 기억(memory)만 주어진 경우에도 결과를 리스트 형태
            + " Memories are separated by semi colans (;)"  # 세미 콜론으로 구분
            + "\Memories: {memory_content}"
            + "\nRating: "
        )
        scores = self.chain(prompt).run(memory_content=memory_content).strip()

        if self.verbose:
            logger.info(f"Importance scores: {scores}")

        # Split into list of strings and convert to floats
        scores_list = [float(x) for x in scores.split(";")]

        return scores_list

    # NPC의 여러 기억들(memories)에 관측한 것과 기억을 추가한다
    # 주어진 기억들을 중요성 점수와 함께 저장하고
    # reflection_threshold 초과 시 반영 작업을 수행하여 NPC의 기억을 강화
    def add_memories(
            self, memory_content: str, now: Optional[datetime] = None
    ) -> List[str]:

        importance_scores = self._score_memories_importance(memory_content)  # 중요성 점수 계산

        self.aggregate_importance += max(
            importance_scores)  # 중요성 점수 중 가장 큰 값 누적하여 합산. 임계값 reflection_treshold 초과시 반영(reflect) 수행하기 위해 사용됨
        memory_list = memory_content.split(";")  # 세미콜론 기준으로 분리하여 리스트
        documents = []

        for i in range(len(memory_list)):  # 기억과 그 기억의 중요성 점수
            documents.append(
                Document(
                    page_content=memory_list[i],
                    metadata={"importance": importance_scores[i]},
                )
            )

        result = self.memory_retriever.add_documents(documents,
                                                     current_time=now)  # NPC 기억저장소인 memory_retriever에 변환된 기억들을 추가

        # 중요성 합(aggregate_importance) < 임계값 reflection_threshold인 경우,
        # 일정량의 기억들을 처리한 후에는, 최근 사건들을 반영(reflect)하여
        # 합성된 새로운 기억을 NPC의 기억 Stream에 추가
        if (
                self.reflection_threshold is not None
                and self.aggregate_importance > self.reflection_threshold
                and not self.reflecting
        ):
            self.reflecting = True
            self.pause_to_reflect(now=now)
            # Hack to clear the importance from reflection
            self.aggregate_importance = 0.0
            self.reflecting = False
        return result

    # NPC의 하나의 기억(memory)에 관측한 것과 기억을 추가한다
    # 일정량 기억들이 처리된 후, 반영(reflect)작업을 수행하여 기억 스트림을 보강
    def add_memory(
            self, memory_content: str, now: Optional[datetime] = None
    ) -> List[str]:

        importance_score = self._score_memory_importance(memory_content)  # 중요성 점수 계산
        self.aggregate_importance += importance_score  # 최근 기억들의 중요성 누적. 임계값 treshold초과 시 반영(reflect) 작업 수행하기 위해 사용
        document = Document(
            page_content=memory_content, metadata={"importance": importance_score}  # Document 객체에 기억과 중요성 점수 저
        )
        result = self.memory_retriever.add_documents([document], current_time=now)  # 추가한 시간과 함께 Document를 저장소에 저장

        if (  # reflection_threshold가 None이고 aggregate_importance보다 작으면, 반영 작업 수행 -> 최근 사건들 반영 및 새로운 합성 기억들이 추가됨
                self.reflection_threshold is not None
                and self.aggregate_importance > self.reflection_threshold
                and not self.reflecting
        ):
            self.reflecting = True
            self.pause_to_reflect(now=now)
            # Hack to clear the importance from reflection
            self.aggregate_importance = 0.0
            self.reflecting = False
        return result

    # 관측과 관련된 기억들을 검색
    # 저장된 기억들 중 주어진 관측과 관련있는 기억들을 찾아 반
    def fetch_memories(
            self, observation: str, now: Optional[datetime] = None  # now = None : 현재 시간
    ) -> List[Document]:

        if now is not None:  # 현재 시간이 아니라면 가장 최근 시간으로 간주
            with mock_now(now):
                return self.memory_retriever.get_relevant_documents(observation)
        else:  # 함수를 호출하는 시점의 현재 시간을 사용
            return self.memory_retriever.get_relevant_documents(observation)

    # Document 객체들의 리스트를 받아 각 기억들을 format하여 하나의 문자열로 반환 : 시간 정보 + 기억 내용
    def format_memories_detail(self, relevant_memories: List[Document]) -> str:
        content = []
        for mem in relevant_memories:
            content.append(self._format_memory_detail(mem, prefix="- "))
        return "\n".join([f"{mem}" for mem in content])

    # 하나의 Document 객체를 받아 해당 기억을 format하여 문자열로 반환 : Document객체 생성 시간 + 기억 내용
    def _format_memory_detail(self, memory: Document, prefix: str = "") -> str:
        created_time = memory.metadata["created_at"].strftime("%B %d, %Y, %I:%M %p")
        return f"{prefix}[{created_time}] {memory.page_content.strip()}"

    # Document 객체들의 리스트를 받아 각 기억의 내용을 하나 문자열로 반환
    def format_memories_simple(self, relevant_memories: List[Document]) -> str:
        return "; ".join([f"{mem.page_content}" for mem in relevant_memories])

    # 최근에 저장된 기억부터 순서대로 토큰을 소비하면서 일정 크기의 토큰 제한을 만족하는 기억들을 반환
    def _get_memories_until_limit(self, consumed_tokens: int) -> str:

        result = []
        for doc in self.memory_retriever.memory_stream[::-1]:
            if consumed_tokens >= self.max_tokens_limit:
                break
            consumed_tokens += self.llm.get_num_tokens(doc.page_content)
            if consumed_tokens < self.max_tokens_limit:
                result.append(doc)
        return self.format_memories_simple(result)

    @property  # memory_variables 메서드를 속성(property)로서 접근
    # 동적으로 로드 될 memory 클래스의 입력 키(input keys)목록 반환
    def memory_variables(self) -> List[str]:
        return []

    # chain에 대한 텍스트 입력 받아 특정 key-value 쌍 반환. chain에 필요한 정보 추
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:

        queries = inputs.get(self.queries_key)
        now = inputs.get(self.now_key)
        if queries is not None:  # 입력 키가 존재하는 경우
            relevant_memories = [  # 관련된 기억들 가져오기 / format 하기
                mem for query in queries for mem in self.fetch_memories(query, now=now)
            ]
            return {
                self.relevant_memories_key: self.format_memories_detail(
                    relevant_memories
                ),
                self.relevant_memories_simple_key: self.format_memories_simple(
                    relevant_memories
                ),
            }

        most_recent_memories_token = inputs.get(self.most_recent_memories_token_key)
        if most_recent_memories_token is not None:  # 해당하는 입력 키가 존재하는 경우
            return {  # 최근 기억들 중 일정 크기의 기억들 가져오기
                self.most_recent_memories_key: self._get_memories_until_limit(
                    most_recent_memories_token
                )
            }
        return {}

    # 기억(memory)에 이 모델 실행의 컨텍스트 저장
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        # inputs : 모델 실행에 사용된 입력 값들 / outputs : 모델 실행 결과들

        # TODO: fix the save memory key 추가 구현 필요
        mem = outputs.get(self.add_memory_key)  # self.add_memory_key에 해당하는 값 확인 후 모델 실행 결과
        now = outputs.get(self.now_key)  # self.now_key에 해당하는 값 확인 후 현재 시간
        if mem:  # 결과가 존재하는 경우
            self.add_memory(mem, now=now)  # 해당 기억을 추가

    # 기억저장소 내용 지우기
    def clear(self) -> None:
        # TODO
        pass
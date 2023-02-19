import torch
import transformers
from datetime import datetime


# tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
# model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")


class QAModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_class = transformers.ElectraForQuestionAnswering
        model_name = "checkpoint-41000"

        # Load the model
        self.model = model_class.from_pretrained(model_name).to(self.device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    def predict(self, question, context):
        # Encode the input text
        encoded_input = self.tokenizer(
            question, context, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        # Generate the answer
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encoded_input)
            # print(outputs)
            answer_start, answer_end = (
                torch.argmax(outputs[0], dim=1).detach().cpu().numpy()[0],
                torch.argmax(outputs[1], dim=1).detach().cpu().numpy()[0],
            )
            # print(f"answer_start: {outputs[0][0][answer_start]}")
            answer = self.tokenizer.decode(
                encoded_input["input_ids"][0][answer_start : answer_end + 1],
                skip_special_tokens=True,
            )
        if answer == "":
            return None
        return {"answer":answer, "probability":float(outputs[0][0][answer_start]) + float(outputs[1][0][answer_end])}

    def predict_long_string(self, question, long_context):
        time = datetime.now()
        result_list = []
        length_limit = 500
        tmp_idx = 0
        if len(long_context) > length_limit:
            for idx, i in enumerate(range(length_limit, len(long_context), length_limit)):
                answer = self.predict(question, long_context[tmp_idx:i])
                tmp_idx = i
                if answer is not None:
                    result_list.append(answer)
        else:
            return [self.predict(question, long_context)]

        print(
            f"Inference time: {datetime.now() - time}, iteration: {len(long_context)//length_limit}"
        )

        return result_list


if __name__ == "__main__":
    qa_model = QAModel()
    # print(qa_model.model(qa_model.tokenizer.encode("안녕하세요", "안녕하세요",return_tensors="pt")))
    print(
        qa_model.predict(
            "삼성전자는 언제 세계 1위를 했는가?",
            "1938년 설립된 삼성물산을 모태로 삼성은 식품과 의복을 주력으로 해 오다가 박정희 대통령과의 회동을 계기로, 1969년 삼성전자를 창립하면서 전자산업에 진출하게 되며 똑같이 전자 사업을 하고 있는[7] LG그룹과 함께 첫 모체 진로그룹의 부도로 위기에 놓인 GTV 인수 물망에 거론됐으나[8] 가격 협상에서 의견차를 좁히지 못해 무산됐다. 이듬해인 1970년 삼성NEC가 설립되어 백색가전 및 AV 기기의 생산이 이루어졌다. 1974년에는 한국반도체를 인수하여 반도체 사업에 진출하였고 1980년에는 한국전자통신을 인수, 그리고 1983년 2월에는 창업주인 이병철 회장이 DRAM 사업에 진출한다는 ‘동경 선언’을 발표하였다. 1983년 미국과 일본에 이어 세계에서 3번째 64K DRAM을 개발하였다. 이때의 메모리 반도체 부문의 투자는 1990년대와 2000년대로 이어지며 지금의 삼성전자 발전 기틀을 잡았다고 평가된다. 삼성은 1990년대까지만 하더라도 재계 상위권에 속하는 대한민국 내 여러 대기업 중 하나에 불과하였다. 그러나 1997년 불어 닥친 경제위기를 계기로 삼성은 광범위한 구조조정을 통해 대한민국 내에서 재계서열 1위의 대기업으로 자리를 굳히게 된다. 이는 당시 경제위기로 대한민국 내 30대 대규모 기업집단 중 16곳이 부도를 맞아 해체된 것과 대비된다. 이후 애플의 아이폰을 필두로 스마트폰 시장이 폭발적으로 확대되자 삼성전자는 소위 패스트 팔로워(Fast Follower) 전략을 사용하여 스마트폰 시장의 강자로 자리매김하게 된다.[9][10] 1980년대~1990년대만 하더라도 삼성전자의 목표는 경쟁사인 일본 기업을 따라잡는 것이었다. 그러나 2010년 삼성의 세계 점유율은 평면 TV와 반도체 메모리에서 1위를 차지하며 모두 일본 업체들을 앞서고 있다. 또한 삼성은 2007년에는 휴대폰 부문에서 모토로라를 누르고 세계 2위의 핸드폰 제조업체에 등재되었다. 스마트 헬스케어 분야에서 새 성장동력을 모색해오던 삼성은 2010년 12월에 메디슨을 인수함으로써 헬스케어사업부문에도 진출하게 되었다. 2009년 스마트폰 시장에도 뛰어들어 갤럭시 라인업을 발표하였으며, 스마트폰 시장에 뛰어든지 2년만인 2011년 3/4분기 스마트폰 세계 1위에 오른다. 삼성전자는, 2012년부터, 노키아와 애플을 제치고, 전체 휴대 전화 점유율 1위, 휴대 전화 부문 매출액 2위를 유지하고 있다. 2013년 2분기 기준으로 스마트폰 부분 영업이익 면에서도 애플을 추월하여 1위를 달성하였다.(SA조사, 2013년 2분기, 삼성 52억불, 애플 46억불) 또한 애플과 삼성을 제외한 다른 휴대폰 회사의 순이익은 삼성과 애플의 1/100도 안되는 수준으로 휴대폰 부분 전체 영업이익의 1%마저도 채 점유하지 못하고 있다. 2013년 판매호조를 보이던 스마트폰 사업은 2014년 들어 급격하게 수익이 악화되었다. 이는 스마트폰 시장이 더이상 성장하지 못하고 있고,[11] 기술의 상향평준화로 샤오미 등 중국 업체들과의 경쟁 또한 심화되고 있기 때문이다. 스마트폰은 삼성전자의 전체 영업이익의 70% 이상을 차지하고 있기 때문에, 이는 삼성전자 전체의 위기로 받아들여지고 있다.[12] 2021년 1분기 전세계 매출 기준 스마트 폰 점유율도 17.5",
        )
    )

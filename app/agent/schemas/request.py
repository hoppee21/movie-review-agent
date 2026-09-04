from pydantic import BaseModel, Field


class RequestAnalysis(BaseModel):
    """只理解用户的问题；电影识别由 MovieAnalysis 单独负责。"""

    question: str = Field(
        min_length=1,
        description="明确重述用户想了解的信息或执行的操作，保留平台、对象、方面和其他限制。"
    )

# Design notes

## Vai trò các agent

- Qwen3-VL: final reasoner
- Gemini / Llama: support agent
- Output của support agent = legal support brief
- Support agent không được:
  - chọn A/B/C/D
  - trả lời trực tiếp câu hỏi
  - kết luận thay Qwen

## Vì sao tách 2 tầng phân loại

1. `dataset_type`:
   sign_identification / rule_compliance / verification / object_presence /
   information_reading / navigation / counting / other

2. `intent`:
   temporal / value / direction / identification / existence / unknown

`dataset_type` quyết định **tri thức pháp luật nào cần bơm**.
`intent` quyết định **cách trình bày support brief**.

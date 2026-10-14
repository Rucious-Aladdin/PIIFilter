# Toy experiment results

- Date: 2026-10-14T09:28:03+00:00
- Detector: `urchade/gliner_multi_pii-v1` (threshold 0.5)
- LLM: `gpt-5.5` (openai backend)
- System prompt: Some personal details in the user's message have been replaced with placeholders such as &lt;NAME_1&gt; or &lt;PHONE_1&gt;. Treat each placeholder as the real value: wherever that detail belongs in your answer, copy the placeholder exactly as written, including the angle brackets.

## Detected spans (GLiNER confidence)

| Example | Gold value | Label | Result | Confidence |
|---|---|---|---|---|
| Example 1 (clean) | Minji Park | NAME | exact | 0.999 |
| Example 1 (clean) | 41 Maple Street | ADDRESS | exact | 0.969 |
| Example 1 (clean) | 010-2345-6789 | PHONE | exact | 1.0 |
| Example 2 (noisy) | minji park | NAME | exact | 0.686 |
| Example 2 (noisy) | 41 maple stret | ADDRESS | exact | 0.921 |
| Example 2 (noisy) | 010 2345 6789 | PHONE | exact | 0.985 |

## Output of each step (Table 9)

| | Example 1 (clean) | Example 2 (noisy) |
|---|---|---|
| User prompt | Write a short message to my landlord. I'm Minji Park from 41 Maple Street, and my heater is broken. He can call me at 010-2345-6789. | write a short message to my landlord im minji park from 41 maple stret and my heater is broken he can call me at 010 2345 6789 |
| Prompt sent to the LLM | Write a short message to my landlord. I'm &lt;NAME_1&gt; from &lt;ADDRESS_1&gt;, and my heater is broken. He can call me at &lt;PHONE_1&gt;. | write a short message to my landlord im &lt;NAME_1&gt; from &lt;ADDRESS_1&gt; and my heater is broken he can call me at &lt;PHONE_1&gt; |
| LLM answer (masked) | Hi, this is &lt;NAME_1&gt; from &lt;ADDRESS_1&gt;. My heater is broken and needs repair. Please call me at &lt;PHONE_1&gt; when you can. Thank you. | Hi, this is &lt;NAME_1&gt; from &lt;ADDRESS_1&gt;. My heater is broken and needs repair. Please call me at &lt;PHONE_1&gt; when you can. Thank you. |
| Answer shown to the user | Hi, this is Minji Park from 41 Maple Street. My heater is broken and needs repair. Please call me at 010-2345-6789 when you can. Thank you. | Hi, this is minji park from 41 maple stret. My heater is broken and needs repair. Please call me at 010 2345 6789 when you can. Thank you. |
| Private? / Enough context? / Usable answer? | Yes / Yes / Yes | Yes / Yes / Yes |

## Check details

**Example 1 (clean)**
- Leaked values: `[]`
- Extra detections: `[]`
- Placeholders missing from the LLM answer: `[]`
- Placeholder-like text left after unmasking: `[]`

**Example 2 (noisy)**
- Leaked values: `[]`
- Extra detections: `[]`
- Placeholders missing from the LLM answer: `[]`
- Placeholder-like text left after unmasking: `[]`


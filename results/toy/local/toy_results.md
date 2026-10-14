# Toy experiment results

- Date: 2026-10-14T09:03:59+00:00
- Detector: `urchade/gliner_multi_pii-v1` (threshold 0.5)
- LLM: `Qwen/Qwen2.5-7B-Instruct` (local backend)
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
| LLM answer (masked) | Subject: Heater Issue<br><br>Dear Landlord,<br><br>I hope this message finds you well. My name is &lt;NAME_1&gt; and I live at &lt;ADDRESS_1&gt;. Unfortunately, I am experiencing issues with my heater and was wondering if you could assist me in getting it repaired. Please feel free to call me at &lt;PHONE_1&gt; at your earliest convenience to discuss further.<br><br>Thank you for your help!<br><br>Best regards,<br>&lt;NAME_1 | Subject: Heater Needs Repair<br><br>Hi [Landlord's Name],<br><br>This is &lt;NAME_1&gt; from &lt;ADDRESS_1&gt;, and I hope this message finds you well. Unfortunately, the heater in my apartment is not working properly, and I was wondering if you could arrange for a repair service at your earliest convenience. I can be reached at &lt;PHONE_1&gt; if you need to discuss this further.<br><br>Thank you for your assistance!<br><br>Best regards,<br>&lt;NAME_1 |
| Answer shown to the user | Subject: Heater Issue<br><br>Dear Landlord,<br><br>I hope this message finds you well. My name is Minji Park and I live at 41 Maple Street. Unfortunately, I am experiencing issues with my heater and was wondering if you could assist me in getting it repaired. Please feel free to call me at 010-2345-6789 at your earliest convenience to discuss further.<br><br>Thank you for your help!<br><br>Best regards,<br>&lt;NAME_1 | Subject: Heater Needs Repair<br><br>Hi [Landlord's Name],<br><br>This is minji park from 41 maple stret, and I hope this message finds you well. Unfortunately, the heater in my apartment is not working properly, and I was wondering if you could arrange for a repair service at your earliest convenience. I can be reached at 010 2345 6789 if you need to discuss this further.<br><br>Thank you for your assistance!<br><br>Best regards,<br>&lt;NAME_1 |
| Private? / Enough context? / Usable answer? | Yes / Yes / No | Yes / Yes / No |

## Check details

**Example 1 (clean)**
- Leaked values: `[]`
- Extra detections: `[]`
- Placeholders missing from the LLM answer: `[]`
- Placeholder-like text left after unmasking: `['NAME_1']`

**Example 2 (noisy)**
- Leaked values: `[]`
- Extra detections: `[]`
- Placeholders missing from the LLM answer: `[]`
- Placeholder-like text left after unmasking: `['NAME_1']`


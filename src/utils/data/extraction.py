def extract_xml_answer(text: str) -> str:
    """Вытаскивает ответ из тегов <output>ответ</output>"""
    answer = text.split("<output>")[-1]
    answer = answer.split("</output>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()
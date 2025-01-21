def get_resources_for_concept(concept_text, concept_list, topic_id):
    clean_concept = concept_text.lower().strip().replace(" ", "")
    matching_concept = next(
        (c for c in concept_list if c['ConceptText'].lower().strip().replace(" ", "") == clean_concept),
        None
    )
    if matching_concept:
        content_payload = {
            'TopicID': topic_id,
            'ConceptID': int(matching_concept['ConceptID'])
        }
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        }
        try:
            response = requests.post(API_CONTENT_URL, json=content_payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching resources: {e}")
            return None
    return None

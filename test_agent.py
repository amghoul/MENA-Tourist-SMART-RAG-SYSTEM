import time
import re
# Import RAGAgent and AgentState from the core logic file
from rag_agent import RAGAgent, AgentState 

# --- TEST SET DEFINITION ---
TEST_CASES = [
    # ------------------
    # ENGLISH TEST CASES (10)
    # ------------------
    {
        "question": "What is the primary function and location of the Al-Azhar Mosque?",
        "expected_answer_keywords": ["Cairo", "Egypt", "Fatimid", "university", "mosque"],
        "language": "English",
        "notes": "Testing a complex entity with multiple attributes."
    },
    {
        "question": "When did the construction of the Hassan II Mosque in Casablanca conclude?",
        "expected_answer_keywords": ["1993", "Casablanca", "Morocco", "minaret"],
        "language": "English",
        "notes": "Testing for date specificity."
    },
    {
        "question": "Who founded the Great Mosque of Kairouan and what century was it established?",
        "expected_answer_keywords": ["Uqba ibn Nafi", "7th century", "Tunisia"],
        "language": "English",
        "notes": "Testing for specific founder and historical timeline."
    },
    {
        "question": "Describe the architectural style of the Citadel of Aleppo in Syria.",
        "expected_answer_keywords": ["fortification", "Aleppo", "Hittite", "medieval"],
        "language": "English",
        "notes": "Testing for architectural and historical context."
    },
    {
        "question": "Which Roman-era ruins are notable in the country of Jordan?",
        "expected_answer_keywords": ["Jerash", "Roman", "Jordan", "columns"],
        "language": "English",
        "notes": "Testing retrieval based on era and location."
    },
    {
        "question": "What role did the Nabataeans play in the development of the city of Petra?",
        "expected_answer_keywords": ["Nabataean", "capital", "trade routes", "water"],
        "language": "English",
        "notes": "Testing for historical role."
    },
    {
        "question": "Where is the Al-Bahrain Fort located and why is it historically important?",
        "expected_answer_keywords": ["Bahrain", "UNESCO", "Dilmun", "Portuguese"],
        "language": "English",
        "notes": "Testing for multiple facts about a single site."
    },
    {
        "question": "What is the key feature of the Mosque of Ibn Tulun in Egypt?",
        "expected_answer_keywords": ["spiral minaret", "Samarra", "Cairo", "Abbasid"],
        "language": "English",
        "notes": "Testing for a specific structural detail."
    },
    {
        "question": "List two major historical sites in Saudi Arabia.",
        "expected_answer_keywords": ["Al-Ula", "Hegra", "Nabataean", "Diriyah"],
        "language": "English",
        "notes": "Testing retrieval of multiple distinct entities."
    },
    {
        "question": "In which city is the ancient souq of Al-Madina located?",
        "expected_answer_keywords": ["Aleppo", "Syria", "covered market", "historical"],
        "language": "English",
        "notes": "Testing location of a non-religious site."
    },

    # ------------------
    # ARABIC TEST CASES (10)
    # ------------------
    {
        "question": "متى تم الانتهاء من بناء قبة الصخرة وأين تقع؟",
        "expected_answer_keywords": ["القدس", "أورشليم", "72", "الأموي"],
        "language": "Arabic",
        "notes": "Testing for location and date using Arabic query."
    },
    {
        "question": "ما هي الأهمية التاريخية لمدينة البتراء في الأردن؟",
        "expected_answer_keywords": ["الأنباط", "البتراء", "عاصمة", "الأردن"],
        "language": "Arabic",
        "notes": "Testing for historical role and Arabic language output fidelity."
    },
    {
        "question": "اذكر اسم مؤسس الجامع الأموي في دمشق.",
        "expected_answer_keywords": ["الوليد بن عبد الملك", "دمشق", "سوريا"],
        "language": "Arabic",
        "notes": "Testing for a proper noun specific to the context."
    },
    {
        "question": "صف القلعة الحمراء في غرناطة من ناحية معمارية.",
        "expected_answer_keywords": ["الأندلس", "غرناطة", "المرينيين", "الحمراء"],
        "language": "Arabic",
        "notes": "Testing architecture description."
    },
    {
        "question": "ما هي المدينة التاريخية الواقعة في اليمن والمشهورة بمبانيها الطينية؟",
        "expected_answer_keywords": ["شبام", "اليمن", "مانهاتن الصحراء", "طينية"],
        "language": "Arabic",
        "notes": "Testing for a specific characteristic and location."
    },
    {
        "question": "اذكر المواقع الأثرية في ليبيا التي تعود إلى العصر الروماني.",
        "expected_answer_keywords": ["لبدة الكبرى", "صبراتة", "الروماني", "ليبيا"],
        "language": "Arabic",
        "notes": "Testing retrieval of multiple names in Arabic."
    },
    {
        "question": "ما هي العلاقة بين واحة العين في الإمارات والتراث العالمي لليونسكو؟",
        "expected_answer_keywords": ["واحة العين", "الإمارات", "اليونسكو", "زراعة"],
        "language": "Arabic",
        "notes": "Testing for relation to UNESCO status."
    },
    {
        "question": "ما هو الجامع الذي يتميز بمئذنته الملوية في العراق؟",
        "expected_answer_keywords": ["سامراء", "العراق", "المئذنة الملوية", "العباسي"],
        "language": "Arabic",
        "notes": "Testing for a specific structural detail and location."
    },
    {
        "question": "ما هو الحدث الرئيسي الذي أدى إلى تدمير أجزاء من مدينة تدمر الأثرية؟",
        "expected_answer_keywords": ["داعش", "سوريا", "تدمر", "تدمير"],
        "language": "Arabic",
        "notes": "Testing for a recent historical event related to a site."
    },
    {
        "question": "أين تقع قلعة صلاح الدين الأيوبي وما هو دورها العسكري؟",
        "expected_answer_keywords": ["القاهرة", "صلاح الدين", "مصر", "عسكري"],
        "language": "Arabic",
        "notes": "Testing location and historical function."
    },
]
# -----------------------------------------------------------------------------------


def check_accuracy_on_test_set(agent: RAGAgent):
    """
    Runs the RAG agent against the predefined TEST_CASES and checks for keyword accuracy.
    
    Args:
        agent (RAGAgent): An initialized instance of the RAGAgent class.
    """
    print("\n" + "=" * 100)
    print("🚀 STARTING ACCURACY TEST AGAINST GOLDEN DATASET (TEST_CASES)")
    print("=" * 100)

    results = []
    start_time = time.time()
    
    # Helper function to check if all keywords are present in the answer
    def check_keywords(answer, keywords):
        # Convert answer to lowercase and remove non-alphanumeric characters for robust checking
        normalized_answer = re.sub(r'[^\w\s]', '', answer).lower()
        
        found_count = 0
        for keyword in keywords:
            # Use a case-insensitive check
            if re.search(re.escape(keyword), answer, re.IGNORECASE) or keyword.lower() in normalized_answer:
                found_count += 1
        
        score = (found_count / len(keywords)) * 100
        return score, found_count

    for i, case in enumerate(TEST_CASES):
        question = case["question"]
        keywords = case["expected_answer_keywords"]
        language = case["language"]
        
        print(f"\n--- TEST CASE {i+1}/{len(TEST_CASES)}: {language} ---")
        print(f"❓ Question: {question}")
        print(f"🔑 Keywords: {keywords}")

        # Run the agent (using the non-streaming ask for simplicity in testing)
        try:
            answer = agent.ask(question)
        except Exception as e:
            answer = f"Error during generation: {e}"

        score, found = check_keywords(answer, keywords)
        status = "✅ PASS" if score == 100.0 else f"⚠️ PARTIAL ({found}/{len(keywords)})"

        print(f"🤖 Answer: {answer[:100]}...")
        print(f"📊 Accuracy: {score:.1f}% ({status})")

        results.append({
            "question": question,
            "language": language,
            "answer": answer,
            "keywords_required": len(keywords),
            "keywords_found": found,
            "accuracy_score": score,
            "status": status
        })

    elapsed_time = time.time() - start_time
    
    # Calculate overall accuracy
    total_score = sum(r['accuracy_score'] for r in results)
    overall_accuracy = total_score / len(TEST_CASES) if results else 0
    
    print("\n" + "=" * 100)
    print("FINAL ACCURACY REPORT")
    print("=" * 100)
    for r in results:
        print(f"[{r['status']}] ({r['language']}) {r['question'][:60]}... -> Score: {r['accuracy_score']:.1f}%")
    
    print("\n" + "=" * 100)
    print(f"📊 OVERALL AVERAGE ACCURACY: {overall_accuracy:.2f}%")
    print(f"⏱️ Total test time: {elapsed_time:.2f} seconds")
    print("=" * 100)
    
    return results, overall_accuracy


if __name__ == "__main__":
    # This block allows running the test script directly, assuming 'rag_agent.py' is in the path
    print("To run the test suite, please execute 'search.py' and type 'test' at the prompt.")
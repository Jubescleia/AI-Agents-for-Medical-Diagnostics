import os
import sys
import io
try:
    from dotenv import load_dotenv
    from pathlib import Path
    env_path = Path(__file__).resolve().parents[1] / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except Exception:
    pass
# Ensure stdout is configured to UTF-8 to avoid Windows cp1252 encoding errors
try:
    # Python 3.7+ has reconfigure
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    # Fallback: wrap the buffer with a UTF-8 text wrapper
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        # Last resort: leave stdout as-is; prints may still fail on some characters
        pass
from langchain_core.prompts import PromptTemplate
from openai import OpenAI

class Agent:
    def __init__(self, medical_report=None, role=None, extra_info=None):
        self.medical_report = medical_report
        self.role = role
        self.extra_info = extra_info
        # Initialize the prompt based on role and other info
        self.prompt_template = self.create_prompt_template()
        # Initialize the OpenAI client with either OPENROUTER_API_KEY or OPENAI_API_KEY
        openrouter_present = bool(os.getenv("OPENROUTER_API_KEY"))
        openai_present = bool(os.getenv("OPENAI_API_KEY"))
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No API key found. Environment presence: OPENROUTER_API_KEY="
                f"{openrouter_present}, OPENAI_API_KEY={openai_present}. "
                "Please set OPENROUTER_API_KEY or OPENAI_API_KEY environment variable (do not paste the key into code)."
            )

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def create_prompt_template(self):
        if self.role == "MultidisciplinaryTeam":
            templates = f"""
                Act like a multidisciplinary team of healthcare professionals.
                You will receive a medical report of a patient visited by a Cardiologist, Psychologist, and Pulmonologist.
                Task: Review the patient's medical report from the Cardiologist, Psychologist, and Pulmonologist, analyze them and come up with a list of 3 possible health issues of the patient.
                Just return a list of bullet points of 3 possible health issues of the patient and for each issue provide the reason.
                
                Cardiologist Report: {self.extra_info.get('cardiologist_report', '')}
                Psychologist Report: {self.extra_info.get('psychologist_report', '')}
                Pulmonologist Report: {self.extra_info.get('pulmonologist_report', '')}
            """
        else:
            templates = {
                "Cardiologist": """
                    Act like a cardiologist. You will receive a medical report of a patient.
                    Task: Review the patient's cardiac workup, including ECG, blood tests, Holter monitor results, and echocardiogram.
                    Focus: Determine if there are any subtle signs of cardiac issues that could explain the patient’s symptoms. Rule out any underlying heart conditions, such as arrhythmias or structural abnormalities, that might be missed on routine testing.
                    Recommendation: Provide guidance on any further cardiac testing or monitoring needed to ensure there are no hidden heart-related concerns. Suggest potential management strategies if a cardiac issue is identified.
                    Please only return the possible causes of the patient's symptoms and the recommended next steps.
                    Medical Report: {medical_report}
                """,
                "Psychologist": """
                    Act like a psychologist. You will receive a patient's report.
                    Task: Review the patient's report and provide a psychological assessment.
                    Focus: Identify any potential mental health issues, such as anxiety, depression, or trauma, that may be affecting the patient's well-being.
                    Recommendation: Offer guidance on how to address these mental health concerns, including therapy, counseling, or other interventions.
                    Please only return the possible mental health issues and the recommended next steps.
                    Patient's Report: {medical_report}
                """,
                "Pulmonologist": """
                    Act like a pulmonologist. You will receive a patient's report.
                    Task: Review the patient's report and provide a pulmonary assessment.
                    Focus: Identify any potential respiratory issues, such as asthma, COPD, or lung infections, that may be affecting the patient's breathing.
                    Recommendation: Offer guidance on how to address these respiratory concerns, including pulmonary function tests, imaging studies, or other interventions.
                    Please only return the possible respiratory issues and the recommended next steps.
                    Patient's Report: {medical_report}
                """
            }
            templates = templates[self.role]
        return PromptTemplate.from_template(templates)
    
    def run(self):
        print(f"{self.role} is running...")
        prompt = self.prompt_template.format(medical_report=self.medical_report)
        try:
            # Allow override of model via environment variable
            model = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")

            completion = self.client.chat.completions.create(
                extra_body={},
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )

            # Safely extract the assistant text from the response
            content = None
            try:
                content = completion.choices[0].message.content
            except Exception:
                # Try alternate common shape
                try:
                    content = completion.choices[0].text
                except Exception:
                    content = str(completion)

            try:
                # test print to stdout encoding by encoding/decoding
                _ = content.encode(sys.stdout.encoding or 'utf-8', errors='strict')
                # If encoding succeeds, return original content
                return content
            except Exception:
                # Fallback: replace characters that can't be encoded so
                # printing doesn't crash the process
                safe = content.encode(sys.stdout.encoding or 'utf-8', errors='replace')
                safe = safe.decode(sys.stdout.encoding or 'utf-8', errors='replace')
                return safe
        except Exception as e:
            print("Error occurred:", e)
            return None

# Define specialized agent classes
class Cardiologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Cardiologist")

class Psychologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Psychologist")

class Pulmonologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Pulmonologist")

class MultidisciplinaryTeam(Agent):
    def __init__(self, cardiologist_report, psychologist_report, pulmonologist_report):
        extra_info = {
            "cardiologist_report": cardiologist_report,
            "psychologist_report": psychologist_report,
            "pulmonologist_report": pulmonologist_report
        }
        super().__init__(role="MultidisciplinaryTeam", extra_info=extra_info)

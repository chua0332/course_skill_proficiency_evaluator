import pandas as pd
import openai
import json
import time
import os
import logging
from dotenv import load_dotenv
from datetime import datetime
from tqdm import tqdm

# ---------------------------------------------------------------------------
# ENVIRONMENT & API SETUP
# ---------------------------------------------------------------------------

load_dotenv('openai.env')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Check your openai.env file.")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

MODEL_NAME          = "gpt-4.1"
INPUT_CSV_PATH      = "./data/extracted_jobs_skills_dataset.csv"
OUTPUT_CSV_PATH     = "./data/jobpostings_skills_with_proficiency_labels.csv"
SAVE_EVERY          = 200        # save the output CSV every 200 rows
MAX_RETRIES         = 4
RETRY_DELAY_SECONDS = 5
TEMPERATURE         = 0.0
MAX_TOKENS          = 300

# ---------------------------------------------------------------------------
# LOGGING — only real errors get logged
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"labelling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert skills taxonomist trained on the Skills Framework 2.0 (SFw2.0) 
Value Creation Principle (VCP). Your task is to classify the proficiency level of 
a skill as it is required in a given job posting, based strictly on how the skill 
is described and applied in the job description.

You must classify the skill as exactly one of: Basic, Intermediate, or Advanced.

Use the following proficiency level definitions, drawn directly from the Skills 
Framework 2.0 Value Creation Principle (Table 1) and its grounding in Bloom's 
Taxonomy and the SOLO Taxonomy.

---

BASIC
Operating context: Work context is likely routine and predictable, and tasks are 
well-defined and specified. Issues are often "Standard" where there is a Standard 
Operating Procedure (SOP) to address it.

Value created: Under predictable and certain operating conditions, individuals are 
required to apply the skill to minimally achieve a specified intended contribution 
to the job function.

The individual is expected to:
- Recall and state facts, methods, processes, and know-how related to the skill
- Explain, interpret, classify, and summarise information
- Apply the skill in familiar contexts by following instructions
- Execute defined tasks and solve routine, well-defined problems
- Make simple connections between concepts without yet seeing deeper relationships

Sample action words from this level:
Memorize, identify, recognize, define, classify, describe, list, discuss, 
illustrate, apply (familiar contexts), explain, summarize, compare, contrast, 
differentiate (straightforward), organize, solve a problem (routine), execute, 
perform, prepare, conduct, operate, support, deploy.

---

INTERMEDIATE
Operating context: Work context is likely less routine and less predictable, and 
tasks are less well-defined. Issues are often "Non-Standard", where there are 
known ways to name and solve the issue even if there is no SOP to address it.

Value created: Under less predictable and less certain operating conditions, 
individuals may require greater technical knowledge and problem-solving skills 
to adapt and respond to developments in work context in order to fulfil the job 
function or attain the business goals.

The individual is expected to:
- Integrate different aspects of the skill and analyse relationships between components
- Understand how parts contribute to the whole
- Execute less defined tasks and solve non-routine, less defined problems
- Monitor, review, and recommend methods to improve task execution
- Substantiate and justify recommended methods and solutions
- Assess and make judgments about the value of information, materials, and methods

Sample action words from this level:
Analyse, predict, conclude, argue, debate, make a case, make a plan, transfer 
(to new contexts), construct, review and rewrite (evaluative), solve a problem 
(non-routine), determine, prioritize (involving judgment), recommend, evaluate,
plan, develop, monitor, implement.

---

ADVANCED
Operating context: Work context is likely non-routine and unpredictable, and tasks 
are likely new, undefined and out of the current job or business context. Issues 
are often highly complex which requires issues to be identified and solutions to 
be developed.

Value created: Under non-routine and unpredictable conditions that may lie outside 
of the individual's existing job or business context, individuals will need to draw 
on knowledge and expertise from other (new) contexts and modify or customise 
application of the skill to create new or better value in order to attain 
"extraordinary" business goals.

The individual is expected to:
- Generalise learning to new domains and hypothesize about new applications
- Execute new and undefined tasks and solve new and undefined problems
- Create original works, methods, or formulate novel solutions
- Synthesise information to influence thinking and push boundaries
- Go beyond immediate job or business contexts to develop overarching principles 
  or new approaches

Sample action words from this level:
Theorize, hypothesize, generalize, reflect, generate, create, compose, invent, 
originate, prove from first principles, make an original case, redesign, innovate,
direct, formulate, drive, establish, lead (in an innovative or strategic capacity).

---

IMPORTANT NOTES:

1. Base your classification on HOW the skill is described and used in the job 
   posting — not on how prestigious or technically complex the skill is in general.
   (e.g. "Python" required to run existing scripts following SOPs = Basic, 
    even though Python is a broad and complex skill.)

2. Pay close attention to the OPERATING CONTEXT described in the job posting — 
   whether tasks are routine and predictable (Basic), less routine requiring 
   judgment (Intermediate), or non-routine requiring creation and innovation 
   (Advanced).

3. If the job posting describes a range of expectations for the skill, classify 
   at the HIGHEST level clearly described.

4. If the job posting is vague and gives no clear signal of how the skill is 
   applied, classify as Basic.

5. Respond ONLY with a JSON object. No explanation, no preamble, no markdown.

---

OUTPUT FORMAT (strict JSON only):
{
  "skillTitle": "<skill title as provided>",
  "proficiencyLevel": "<Basic | Intermediate | Advanced>",
  "rationale": "<1-2 sentences citing specific language from the job description that supports your classification>"
}"""


def build_user_prompt(job_title: str, job_description: str, skill_title: str) -> str:
    return f"""Classify the proficiency level of the following skill based on how it is required and applied in the job posting below.

Job Title: {job_title}

Job Description: {job_description}

Skill Title: {skill_title}

Respond only with the JSON output as specified. Do not include any text outside the JSON."""


# ---------------------------------------------------------------------------
# CORE LABELLING FUNCTION
# ---------------------------------------------------------------------------

def label_single_row(job_title: str, job_description: str, skill_title: str) -> dict:
    """
    Calls ChatGPT to label one (job posting, skill) pair.
    On success: returns proficiencyLevel, rationale, raw_response, error=None
    On failure: returns all None except error which contains the error message
    """
    user_prompt = build_user_prompt(job_title, job_description, skill_title)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )

            raw_text = response.choices[0].message.content.strip()

            # Strip markdown code fences if model wraps response in ```json...```
            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`").strip()
                if raw_text.lower().startswith("json"):
                    raw_text = raw_text[4:].strip()

            parsed = json.loads(raw_text)

            level = parsed.get("proficiencyLevel", "").strip()
            if level not in ("Basic", "Intermediate", "Advanced"):
                raise ValueError(f"Unexpected proficiencyLevel value: '{level}'")

            return {
                "proficiencyLevel": level,
                "rationale":        parsed.get("rationale", ""),
                "raw_response":     raw_text,
                "error":            None
            }

        except json.JSONDecodeError as e:
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES} - JSON parse error: {e}")
        except ValueError as e:
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES} - Validation error: {e}")
        except openai.RateLimitError:
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES} - Rate limit hit, waiting 60s...")
            time.sleep(60)
        except openai.APIError as e:
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES} - API error: {e}")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS)

    logger.error(f"FAILED: skill='{skill_title}' | job title='{job_title}'")
    return {
        "proficiencyLevel": None,
        "rationale":        None,
        "raw_response":     None,
        "error":            f"Failed after {MAX_RETRIES} attempts"
    }


# ---------------------------------------------------------------------------
# MAIN LABELLING PIPELINE
# ---------------------------------------------------------------------------

def run_labelling_pipeline(input_csv: str, output_csv: str) -> pd.DataFrame:

    # --- Load and validate input ---
    print(f"Loading data from: {input_csv}")
    df = pd.read_csv(input_csv)

    required_cols = {"job_id", "jobTitle", "jobDescription", "skillTitle"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    print(f"{len(df)} rows loaded | {df['job_id'].nunique()} unique job IDs | {df['jobTitle'].nunique()} unique job titles | {df['skillTitle'].nunique()} unique skills\n")

    # --- Initialise the 4 output columns as None before we start ---
    df["gpt_proficiencyLevel"] = None
    df["gpt_rationale"]        = None
    df["gpt_raw_response"]     = None
    df["gpt_error"]            = None

    # --- Main labelling loop ---
    for i, idx in enumerate(tqdm(df.index, desc="Labelling")):
        row = df.loc[idx]

        result = label_single_row(
            job_title=str(row["jobTitle"]),
            job_description=str(row["jobDescription"]),
            skill_title=str(row["skillTitle"])
        )

        # Write result back into the dataframe row
        df.at[idx, "gpt_proficiencyLevel"] = result["proficiencyLevel"]
        df.at[idx, "gpt_rationale"]        = result["rationale"]
        df.at[idx, "gpt_raw_response"]     = result["raw_response"]
        df.at[idx, "gpt_error"]            = result["error"]

        # Save the whole dataframe to CSV every SAVE_EVERY rows
        if (i + 1) % SAVE_EVERY == 0:
            df.to_csv(output_csv, index=False)
            print(f"  Progress saved at row {i + 1}")

    # --- Summary before final save ---
    labelled_df = df[df["gpt_proficiencyLevel"].notna()]
    print(f"\n  Unique job IDs labelled    : {labelled_df['job_id'].nunique()}")
    print(f"  Unique job titles labelled : {labelled_df['jobTitle'].nunique()}")
    print(f"  Unique skills labelled     : {labelled_df['skillTitle'].nunique()}")

    # --- Final save after loop completes ---
    df.to_csv(output_csv, index=False)

    # --- Summary ---
    labelled = df["gpt_proficiencyLevel"].notna().sum()
    failed   = df["gpt_error"].notna().sum()
    dist     = df["gpt_proficiencyLevel"].value_counts().to_dict()

    print(f"\nDone! Output saved to: {output_csv}")
    print(f"  Labelled     : {labelled}")
    print(f"  Failed       : {failed}")
    print(f"  Distribution : {dist}")

    return df


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result_df = run_labelling_pipeline(
        input_csv=INPUT_CSV_PATH,
        output_csv=OUTPUT_CSV_PATH
    )

    preview_cols = [
        "job_id", "jobTitle", "skillTitle",
        "gpt_proficiencyLevel", "gpt_rationale"
    ]
    print("\nSample output:")
    print(result_df[preview_cols].head(10).to_string(index=False))
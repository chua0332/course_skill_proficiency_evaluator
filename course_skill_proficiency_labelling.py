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
INPUT_CSV_PATH      = "./data/final_course_skills_merged_modified_algo.csv"
OUTPUT_CSV_PATH     = "./data/courses_skills_with_proficiency_labels_modified_algo.csv"
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

SYSTEM_PROMPT = """You are an expert skills taxonomist. Your task is to classify the proficiency 
level at which a given skill is taught in a course, based strictly on the depth 
and cognitive nature of learning described in the course description.

You must classify the skill as exactly one of: Basic, Intermediate, or Advanced.

Use the following proficiency level definitions, which are drawn from the 
Skills Framework 2.0 (SFw2.0) Value Creation Principle and its grounding in 
Bloom's Taxonomy and the SOLO (Structure of the Observed Learning Outcome) 
Taxonomy.

---

BASIC
The course teaches the skill at the level of knowing, comprehending, and applying 
in familiar, defined contexts.

Learners are expected to:
- Recall and state facts, methods, processes, and know-how related to the skill
- Explain information in their own words, interpret, classify, and summarise
- Execute defined tasks by following instructions in straightforward, familiar contexts
- Solve defined, routine problems
- Make simple connections between concepts but not yet deeper relational connections

Sample action words from this level:
Memorize, identify, recognize, define, classify, describe, list, discuss, 
illustrate, apply (familiar contexts), explain, summarize, compare, contrast, 
differentiate (straightforward), organize, solve a problem (routine), execute, 
perform, prepare.

---

INTERMEDIATE
The course teaches the skill at the level of analysing, evaluating, and exercising 
judgment in less defined, less familiar contexts.

Learners are expected to:
- Integrate different aspects of the skill and analyse relationships between components
- Understand how parts contribute to a whole
- Solve less defined, non-routine problems involving a degree of judgment
- Monitor, review, recommend, and substantiate methods and solutions
- Make connections across components and assess the value of information and methods

Sample action words from this level:
Analyse, predict, conclude, argue, debate, make a case, make a plan, transfer 
(to new contexts), construct, review and rewrite (evaluative), solve a problem 
(non-routine), determine, prioritize (involving judgment), recommend, evaluate.

---

ADVANCED
The course teaches the skill at the level of creating, innovating, and generating 
novel solutions, going beyond existing contexts and established approaches.

Learners are expected to:
- Generalise learning to new domains and hypothesize about new applications
- Create original works, methods, or formulate novel solutions
- Synthesise information to influence thinking and present new ideas
- Solve new and undefined problems from first principles
- Go beyond immediate contexts to develop overarching principles or new approaches

Sample action words from this level:
Theorize, hypothesize, generalize, reflect, generate, create, compose, invent, 
originate, prove from first principles, make an original case, redesign, innovate.

---

IMPORTANT NOTES:

1. Base your classification on the COGNITIVE DEPTH described in the course 
   description for the given skill — not on how complex or prestigious the 
   skill is in general.

2. The levels are progressive and build on one another. A learner classified 
   at Advanced is assumed to also be competent at Intermediate and Basic.

3. If the course description covers a range of depths, classify at the HIGHEST 
   cognitive level clearly described.

4. If the course description is vague and gives no clear signal of cognitive 
   depth, classify as Basic.

5. Respond ONLY with a JSON object. No explanation, no preamble, no markdown.

---

OUTPUT FORMAT (strict JSON only):
{
  "skillTitle": "<skill title as provided>",
  "proficiencyLevel": "<Basic | Intermediate | Advanced>",
  "rationale": "<1-2 sentences citing specific language from the course description that supports your classification>"
}"""


def build_user_prompt(course_title: str, course_description: str, skill_title: str) -> str:
    return f"""Classify the proficiency level of the following skill based on how it is covered in the course description below.

Course Title: {course_title}

Course Description: {course_description}

Skill Title: {skill_title}

Respond only with the JSON output as specified. Do not include any text outside the JSON."""


# ---------------------------------------------------------------------------
# CORE LABELLING FUNCTION
# ---------------------------------------------------------------------------

def label_single_row(course_title: str, course_description: str, skill_title: str) -> dict:
    """
    Calls ChatGPT to label one (course, skill) pair.
    On success: returns proficiencyLevel, rationale, raw_response, error=None
    On failure: returns all None except error which contains the error message
    """
    user_prompt = build_user_prompt(course_title, course_description, skill_title)

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

    logger.error(f"FAILED: skill='{skill_title}' | course='{course_title}'")
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

    required_cols = {"coursetitle", "coursedescription", "skillTitle", "coursereferencenumber"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    print(f"{len(df)} rows loaded | {df['coursereferencenumber'].nunique()} unique courses | {df['skillTitle'].nunique()} unique skills\n")

    # --- Initialise the 4 output columns as None before we start ---
    df["gpt_proficiencyLevel"] = None
    df["gpt_rationale"]        = None
    df["gpt_raw_response"]     = None
    df["gpt_error"]            = None

    # --- Main labelling loop ---
    for i, idx in enumerate(tqdm(df.index, desc="Labelling")):
        row = df.loc[idx]

        result = label_single_row(
            course_title=str(row["coursetitle"]),
            course_description=str(row["coursedescription"]),
            skill_title=str(row["skillTitle"])
        )

        # Write result back into the dataframe row
        df.at[idx, "gpt_proficiencyLevel"] = result["proficiencyLevel"]
        df.at[idx, "gpt_rationale"]        = result["rationale"]
        df.at[idx, "gpt_raw_response"]     = result["raw_response"]
        df.at[idx, "gpt_error"]            = result["error"]

        # Save the whole dataframe to CSV every SAVE_EVERY rows
        # This means if the script crashes, you lose at most SAVE_EVERY rows of work
        if (i + 1) % SAVE_EVERY == 0:
            df.to_csv(output_csv, index=False)
            print(f"  Progress saved at row {i + 1}")
            
    # --- Summary before final save ---
    labelled_df = df[df["gpt_proficiencyLevel"].notna()]
    print(f"Unique courses labelled: {labelled_df['coursereferencenumber'].nunique()}")
    print(f"Unique skills labelled:  {labelled_df['skillTitle'].nunique()}")

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
        "coursereferencenumber", "coursetitle", "skillTitle",
        "gpt_proficiencyLevel", "gpt_rationale"
    ]
    print("\nSample output:")
    print(result_df[preview_cols].head(10).to_string(index=False))
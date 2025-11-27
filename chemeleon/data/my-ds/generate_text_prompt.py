import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import dotenv
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential
from pymatgen.core import Composition

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

dotenv.load_dotenv()

# Retry function for API calls
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def process_row(row, llm_chain, prompt_dir):
    prompt_data = row.to_dict()
    # Run the LLMChain
    chat_val = llm_chain.run(prompt_data)

    # Write output to file
    output_file = prompt_dir / f"{row['material_id']}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(chat_val)


def main(data_dir: str = "."):
    data_dir = Path(data_dir)
    prompt_dir = data_dir / "prompts"
    prompt_dir.mkdir(exist_ok=True, parents=True)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set in your environment variables.")

    # Load dataset
    df_total = pd.read_csv(data_dir / "val.csv")
    print(f"Total crystal text: {len(df_total)}")

    # Skip already generated prompts
    already_prompt = [p.stem for p in prompt_dir.glob("*.txt")]
    print(f"Already prompted: {len(already_prompt)}")
    df_total = df_total[~df_total["material_id"].isin(already_prompt)]
    print(f"Remaining crystal text: {len(df_total)}")

    # Add reduced formula column
    df_total["reduced_formula"] = df_total["composition"].apply(
        lambda x: Composition(x).reduced_formula
    )

    print("Creating LLMChain...")

    # Initialize LLM
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-3.5-turbo",
        temperature=0
    )

    # Define prompt template
    template = """Provide five concise captions for "{reduced_formula}, {crystal_system}"

Here are some examples for other crystal systems:
1. Orthorhombic crystal structure of ZnMnO4
2. Crystal structure of LiO2 in orthorhombic symmetry
3. Cubic symmetry in SiC crystal structure

Please provide five captions for the crystal structure of {reduced_formula} in {crystal_system} symmetry.
"""
    prompt = PromptTemplate(
        input_variables=["reduced_formula", "crystal_system"],
        template=template
    )

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # Process rows with ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_row, row, llm_chain, prompt_dir)
            for _, row in df_total.iterrows()
        ]

        # Show progress with tqdm
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing rows"
        ):
            try:
                future.result()
            except Exception as exc:
                print(f"Task generated an exception: {exc}")


if __name__ == "__main__":
    main()

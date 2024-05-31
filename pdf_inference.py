# %%
import pymupdf  # PyMuPDF   
from bs4 import BeautifulSoup
import pickle
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import LoraConfig

torch.random.manual_seed(0)

model_kwargs = dict(
    use_cache=False,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct", **model_kwargs)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

adapter_model_name = "./checkpoint_dir"

peft_config = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": "all-linear",
    "modules_to_save": None,
}
peft_pdf_parser = LoraConfig(**peft_config)
#model.add_adapter( peft_pdf_parser,adapter_name="pdf_parser")
model.load_adapter(adapter_model_name, adapter_name="pdf_parser")
model.set_adapter('pdf_parser')

SYSTEM = '''The user's input is data in XML format. Please organize it into a markdown format. Pay attention to:

1. Directly output the results.
2. Other than the <p>n </p>, Do not alter any text from the XML. Do not change number into words.
3. Correct format errors, such as misalignment between numbers and text, and disorder in the sequence of table cells.
4. Use markdown, but all numbers must be explicitly written out in full (e.g., 3.2.5.1).
5. Preserve the original document structure as much as possible, such as paragraphs, lists, etc.
6. Pay attention to detecting tables in the text (as the table format may have been lost due to copying from the XML). Restore the table's format and maintain its integrity. Some tables may be too long and span across pages. Pay attention to merging the same tables that span pages. Properly handle table headers to avoid repetition or omission.
7. Text from the XML may contain some garbled characters; remove any characters that are garbled.
8. Convert headings (H1, H2, H3, etc.) into their respective Markdown heading levels (e.g., 3 for # 3, 3.2 for ## 3.2, 3.2.1 for ### 3.2.1).
9. Include metadata information in the output, such as document title, section number, etc. 
10. Remove the footnote and page number, it is important!!!
11. Make sure phrase connected with - will not break up.
'''

def merge_elements_up_to_max_length(elements, max_length):
    """
    Merge elements in the list to ensure no element exceeds the specified max_length.
    
    Parameters:
    - elements: List[str] - The list of string elements to merge.
    - max_length: int - The maximum allowed length for any element after merging.
    
    Returns:
    - List[str]: A new list where the elements have been merged as necessary.
    """
    if not elements:
        return []

    # Initialize the list with the first element
    merged = [elements[0]]

    for element in elements[1:]:
        # Check if the last element in merged list can be combined with the current element
        if len(merged[-1]) + len(element) <= max_length:
            merged[-1] += element  # Merge with the last element
        else:
            merged.append(element)  # Add as a new element

    return merged


pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 2000,
    "return_full_text": False,
    #"temperature": 0.0,
    "do_sample": False,
}

# %%
filename ='2024022700210.pdf'
elements=[]
with pymupdf.open(filename) as doc:
    for page in doc:
        soup = BeautifulSoup(page.get_text('xhtml'), 'html.parser')
        for img in soup("img"):
            img.decompose()

        for item in soup.find_all('p'):
            if len(item.get_text())<2:
                item.decompose()
            else:
                elements.append(str(item))
        elements.append("<hr>")

max_length=7300

merged_elements=merge_elements_up_to_max_length(elements, max_length)

markdown_pairs=[]
for j in range(len(merged_elements)):
    item =merged_elements[j]
    messages=[{"role": "system", "content": SYSTEM},
                {"role": "user", "content": item}]
    output = pipe(messages, **generation_args)
    markdown_pairs.append({'html':item,'markdown':output[0]['generated_text']})

file = open(filename[:-4]+'.pickle', 'wb')
# dump information to that file
pickle.dump(markdown_pairs, file)
# close the file
file.close()



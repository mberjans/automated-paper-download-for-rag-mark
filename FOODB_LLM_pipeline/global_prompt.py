# Maximum number of tokens the model can generate
global_output_max_new_tokens = 2048

# Prompt to instruct the model to generate simple sentences
simple_sentence_prompt = """ Convert the given scientific text into a list of simple, clear sentences. Each sentence should:
1. Express a single fact or relationship
2. Be grammatically complete and independent
3. Use clear, direct language with subject-verb-object structure where possible
4. Remove unnecessary complexity and ambiguity
5. Preserve all important information from the original text
6. Include  specific scientific measurements, names, numbers, units, and values when present
7. Replace pronouns with specific subject names or terms where possible.
8. Format each simplified sentence on a new line.
9. Preserve scientific terminology and exact compound names,
10. For tables and structured data, create sentences that preserve the relationships between columns.
11. preserve full meaning of acronyms and abbreviations.

Example:
Input: "Agent X shows anticancer effects against breast cancer in MCF-7 cells with IC50 of 25μM"
### Response Simplified:
"Agent X shows anticancer effects against breast cancer."
"Agent X is effective in MCF-7 cells." 
"Agent X has IC50 of 25μM in MCF-7 cells."

Input: "Spinach and beetroot are rich in iron and vitamin C. Both are used to alleviate iron deficiency anaemia."
### Response Simplified:
"Spinach is rich in iron."
"Spinach is rich in vitamin C."
"Spinach helps alleviate iron deficiency anaemia."
"Beetroot is rich in iron."
"Beetroot is rich in vitamin C."
"Beetroot helps alleviate iron deficiency anaemia."

Input: "Tomatoes contain lycopene, which has been shown to reduce cancer risk. It is used in many culinary dishes."
### Response Simplified:
"Tomatoes contain lycopene."
"Lycopene reduces cancer risk."
"Tomatoes are used in culinary dishes."

Input: "LCMS Q-TOF analysis showed that catechin content was 2.79-7.53 mg/100g and epicatechin was 27.46-56.84 mg/100g."
### Response Simplified:
"LCMS Q-TOF analysis was used to measure catechin content."
"Catechin content ranged from 2.79 to 7.53 mg per 100g."
"Epicatechin content ranged from 27.46 to 56.84 mg per 100g."

Input: Finally, it is to be noted that the extraction method had a significant effect on the extracted polysaccharides structural features.
For instance, Zhu et al. 35 findings revealed that M. charantia water-extracted poly-saccharides (WBPS) and alkali-extracted polysaccharides (ABPS) pre-sented significant differences in their uronic acid contents (22.23 and 5.69), monosaccharide composition, and molecular weight (Mw: 332 kDa and 1552 kDa), respectively.
While water constitutes the major part of the fruit (93.2), polysaccharides and proteins account for 30 and 18.02 of their dried weight, respectively 32.

### Response Simplified:
The extraction method has a significant effect on the structural features of extracted polysaccharides.
Findings revealed differences between water-extracted polysaccharides (WBPS) and alkali-extracted polysaccharides (ABPS) from *M. charantia*.
water-extracted polysaccharides (WBPS) have an uronic acid content of 22.23%.
alkali-extracted polysaccharides (ABPS) have an uronic acid content of 5.69%.
water-extracted polysaccharides (WBPS) have a molecular weight (Mw) of 332 kDa.
alkali-extracted polysaccharides (ABPS) have a molecular weight (Mw) of 1552 kDa.
*M. charantia* fruit contains 93.2% water. 
Polysaccharides account for 30% of the dried weight of *M. charantia* fruit.
Proteins account for 18.02% of the dried weight of *M. charantia* fruit.

Only use facts found explicitly in the input text. Do not invent, assume, or add any new information.
Now convert the Input text to simple sentences following those guidelines.
Input:  

### Response:
"""

food_entity_prompt = """Below is an instruction that describes a task, 

    ### Instruction: Using the sample_input, carefully extract food entities that are edible food names and specific, identifiable food compounds, that can serve as the subject and object in a subject, predicate, object relationship.

     ### INCLUSION CRITERIA:
    Include the following types of entities:
    1. **Specific foods** — Common or scientific food names (e.g., "green tea", "melon soup", "grape skin extract").
    2. **Food constituents** — Named bioactive compounds found in foods (e.g., "epicatechin", "curcumin", "resveratrol", "EGCG").

    ### EXCLUSION RULES:
    - ❌ Exclude plant parts **not tied to specific food names** (e.g., "leaves", "seeds", "fruit", "pulp"), (e.g. **Mango fruit NOT JUST fruit**)..
    - ❌ Exclude vague or group-level chemical categories (e.g., "polyphenols", "flavonoids", "bioactive compounds" OR *food*, *diet*, *consumption*).
    - Only include specific, identifiable food compounds, such as *catechin*, *epicatechin*, *gallic acid*, or compounds with **precise IUPAC names that can be linked to a ChEBI ID or PubChem ID**.
    - ❌ Exclude **disease names**, **enzymes**, **reagents**, or **hormones** not eaten as food.
    - ❌ Exclude general or vague references (e.g., "diet", "consumption", "food").
    - ❌ Exclude duplicates or plural forms - (convert to singular form)..
    - ❌ Exclude phrases like "green tea consumption" — only extract the food or compound itself.
    - ✅ Expand acronyms to full names where applicable.
    - ✅ Include entities regardless of grammatical role.
    
    Return your output as a **JSON string only**, with a single key `"entities"` containing a list of entity strings.
    If no entities are found, return: `{"entities": []}`.
    **Do not add any other text or explanation.**
    **Do not add any other text like "```json" to it**.  
    ### Few-Shot Examples:

    #### Example 1:
    sample_input:
    The antioxidant activity of epicatechin and catechin isolated from green tea has been widely studied. Unlike general polyphenols, these specific flavan-3-ols demonstrate enhanced free radical scavenging activity.

    Output:
    {"entities": ["epicatechin", "catechin", "green tea"]}

    #### Example 2:
    sample_input:
    Traditional dishes like melon soup made with egusi seeds and bitter leaf are common in West Africa. Studies also noted the presence of gallic acid in grape skin extract.

    Output:
    {"entities": ["melon soup", "egusi seeds", "bitter leaf", "gallic acid", "grape skin extract"]}

    ### sample_input:
    {input:}

    ##
    Response:
    """

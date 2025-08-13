# CodeAgent
Help To Automate Code For Your Projects Using LLM 


### Install Dependency 

```bash
 pip install -r requirements.txt
```

### Initize Agent Using Proplexity API

```python
from CodeAgent  import CodeAgent
from CodeAgentV2  import CodeAgent 
from CodeAgentV3  import CodeAgent

agent = CodeAgent("<enter apikey>")

```

### Just Generate Using Prompt

```python
agent.generate(
    "Explain About Artificial Intelligence"
).json()

```

### To Automate Flow - Example Project


prompt = """
 You are Ai Agent . You will able to code like ai research scientist

 Code For SmolAgents

Instructions :

  1) Agent should answer tech related questions
  2) Execution not supported
  3) Give only Code
  4) python only support
  5) Should Consider doc string


 user:  Building embedding model using contrastive learning . MultiModal embedding model ( Image + Text)

 Dataset Link and Description :
   Kaggle creadiential also settled up and

   load dataset


    ```python
    !mkdir -p /root/.kaggle
    !cp kaggle.json /root/.kaggle
    !chmod 600 /root/.kaggle/kaggle.json
    !kaggle datasets download paramaggarwal/fashion-product-images-small

    ```


  dataset load using python :

  import pandas as pd

  df = pd.read_csv("/content/myntradataset/styles.csv",on_bad_lines="skip")
  df.head()

  id	gender	masterCategory	subCategory	articleType	baseColour	season	year	usage	productDisplayName
  0	15970	Men	Apparel	Topwear	Shirts	Navy Blue	Fall	2011.0	Casual	Turtle Check Men Navy Blue Shirt
  1	39386	Men	Apparel	Bottomwear	Jeans	Blue	Summer	2012.0	Casual	Peter England Men Party Blue Jeans
  2	59263	Women	Accessories	Watches	Watches	Silver	Winter	2016.0	Casual	Titan Women Silver Watch
  3	21379	Men	Apparel	Bottomwear	Track Pants	Black	Fall	2011.0	Casual	Manchester United Men Solid Black Track Pants
  4	53759	Men	Apparel	Topwear	Tshirts	Grey	Summer	2012.0	Casual	Puma Men Grey T-shirt


  Use HuggingfacePretrained Bert and VIT model for embedding model and Use Torch and langchain(if needed)

  Embedding Model Train Using Contrastive Learning with Evaluation and Testing

  Save Best Model and load it for inference Then Save Log File also

  Progress bar using tqdm and cuda support

  Give me a full final code



"""

### To Start Runing And Bebugging

```python 

agent(prompt)

```

*** outputs are Stored Local Folders ***

import importlib
import BERT_explainability.modules.BERT.RoBERTa
import BERT_explainability.modules.BERT.RobertaForSequenceClassification
importlib.reload(BERT_explainability.modules.BERT.RoBERTa)
importlib.reload(BERT_explainability.modules.BERT.RobertaForSequenceClassification)
import torch

from BERT_explainability.modules.BERT.RobertaForSequenceClassification import RobertaForSequenceClassification
model = RobertaForSequenceClassification.from_pretrained('roberta-base')

from transformers import AutoTokenizer
from BERT_explainability.modules.BERT.ExplanationGenerator import Generator
model.eval()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
explanations = Generator(model)

labels_list = ['Corporate',
            'Crime',
            'Cybersecurity',
            'Economics',
            'Environment',
            'Geopolitics',
            'Healthcare',
            'Infrastructure, Transportation and Energy',
            'Military',
            'Politics',
            'Terrorism']

def visualize_explaination_model(model, tokenizer, input_text):
  tokenized_input = tokenizer(input_text, return_tensors='pt',padding='max_length',truncation=True)
  tokenized_input.to(device)

  input_ids = tokenized_input['input_ids']
  attention_mask = tokenized_input['attention_mask']

  # generate an explanation for the input
  expl = explanations.generate_LRP(input_ids=input_ids, attention_mask=attention_mask, start_layer=0)[0]
  # normalize scores
  expl = (expl - expl.min()) / (expl.max() - expl.min())

  # get the model classification
  output = torch.sigmoid(model(input_ids=input_ids, attention_mask=attention_mask)[0])
  print(output)
  predictions = [[labels_list[i] for i in range(len(prediction)) if prediction[i] >= 0.5] for prediction in output]

  print(predictions)

  tokens = tokenizer.convert_ids_to_tokens(input_ids.flatten())
  vis_data_records = [visualization.VisualizationDataRecord(
                                  expl,
                                  0,
                                  '',
                                  '',
                                  '',
                                  1,       
                                  tokens,
                                  1)]
  visualization.visualize_text(vis_data_records)

  return input_ids, expl

text_input="""Herschel Walker, Republican candidate for U.S. Senate in Georgia, walks off-stage during pre-race ceremonies at a NASCAR event in Hampton, Georgia, on Sunday.\nProblem-plagued U.S. Senate candidate Herschel Walker has presented an astounding argument for not enacting laws against air pollution: America’s “good air” will simply “decide” to go to China, he told supporters in Georgia.\n“Since we don’t control the air, our good air decided to float over to China’s bad air, so when China gets our good air, their bad air got to move,” Walker explained. “So it moves over to our good air space. Then now we got to clean that back up,” he added.\nHerschel on the climate/Green New Deal/air:\nThey make bad air and export it to the US for cleaning. After we’ve cleaned their air we send back the good air and they send us another order of bad air. Econ 101, really.\nSweet Jesus, Georgia. This isn’t difficult.\nJesus. This is the @GOP Georgia primary winner and their chosen candidate for the United States Senate. The Senate. Our Senate. One of one-hundred senators.\nNot sending their best.\n"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_ids, expl = visualize_explaination_model(model, tokenizer, text_input)


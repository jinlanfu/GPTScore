from gpt_inference import GPT3Model

def gpt3score(input, output,gpt3model=None,api_key=None):
    gpt3model_name = ''
    if gpt3model == 'ada':
        gpt3model_name = "text-ada-001"
    elif gpt3model == 'babbage':
        gpt3model_name = "text-babbage-001"
    elif gpt3model == 'curie':
        gpt3model_name = "text-curie-001"
    elif gpt3model == 'davinci001':
        gpt3model_name = "text-davinci-001"
    elif gpt3model == 'davinci003':
        gpt3model_name = "text-davinci-003"
    print('gpt3model_name: ', gpt3model_name)

    # "text-curie-001", "text-ada-001", "text-babbage-001", "text-davinci-001", "text-davinci-003",
    metaicl_model = GPT3Model(gpt3model_name, api_key)
    avg_loss = metaicl_model.do_inference(input, output)
    score = -avg_loss
    return score
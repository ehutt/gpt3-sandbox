import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from api import demo_web_app
from api import GPT, Example, UIConfig

import tqdm
import random
from transcript_utils import get_args, \
    get_secret_key, \
    save_transcript, \
    load_faqs, \
    load_squad_queries


def main():
    args = get_args()
    get_secret_key()

    # feed examples
    gpt = prime_model()

    # set generic greeting
    greeting = f"Agent: Hi, thank you for calling the {args.kb_name} Help Center, my name is Jane. " \
               "How may I help you today?\n"

    # test
    test(gpt, greeting)

    # load questions
    if '.json' in args.query_path:
        queries, = load_squad_queries(args.query_path)
    elif '.csv' in args.query_path:
        queries, = load_faqs(args.query_path)
    else:
        raise ValueError('Supported query file types are .json (squad) and .csv (faq).')

    random.shuffle(queries)
    print('Loaded queries.')
    output_path = args.output_path
    i = len(os.listdir(output_path))  # start count at length of dir so as not to override existing transcripts
    for q in tqdm.tqdm(queries[0:20]):
        print('Q{}: {}'.format(i, q))
        i += 1
        prompt = format_prompt(greeting, q)
        transcript = get_transcript(gpt, prompt)
        save_transcript(output_path, i, transcript)


def format_prompt(greeting, query):
    return '{}Caller: Hi. {}\nAgent:'.format(greeting, query)


def prime_model():
    # Construct GPT object and show some examples
    gpt = GPT(engine="davinci",
              temperature=0.9,
              max_tokens=600
              )

    walmart_prompt1 = "Agent: Hi, thank you for calling the Walmart Help Center, my name is Jane. " \
                      "How may I help you today?\nCaller: Hi, yeah. I didn't get any emails for pick up?\n"
    walmart_example1 = "\nAgent: Okay, I can help you with that. First, can I please get your name?" \
                       "\nCaller: It's John. Do you want my email or anything?" \
                       "\nAgent: Thanks John. If you have the order number handy, then I can check on it right " \
                       "now for you. An email address would also be helpful." \
                       "\nCaller: Okay sure, it's order number 12345678 and my email is john@gmail.com" \
                       "\nAgent: Okay, one moment while I look that up for you." \
                       "\nCaller: Sure. Yeah I just want to see if it's ready to be picked up, " \
                       "but also it's kinda weird that I didn't get any emails about it." \
                       "\nAgent: Hi John, thanks for your patience. I can see here that your order isn't ready for " \
                       "pickup yet, but it should be in the next couple of hours. It's possible that the emails " \
                       "got sent to your email spam folder, so that could be worth checking. I'm sorry for " \
                       "the inconvenience. If you'd like, I can have the store give you a call when the order " \
                       "is ready instead." \
                       "\nCaller: Oh, that would be great." \
                       "\nAgent: Of course. What phone number would you like us to use?" \
                       "\nCaller: Sure it's uh 123-456-7890" \
                       "\nAgent: Thank you, you should be receiving a call sometime in the next few hours when your " \
                       "order is ready for pickup. Is there anything else I can help you with today?" \
                       "\nCaller: No, that's it. Thanks." \
                       "\nAgent: My pleasure. Thank you for calling Walmart, you have a great day."

    ciscoit_prompt1 = "Agent: Sorry, you're speaking with Jane how may I assist you?" \
                      "\nCaller: Hi, I'm gonna set up my email. I'm on mobile. I got a new phone and, uh, " \
                      "I'm following the stuff, but the. Um, I got to the E store part downloaded E store. " \
                      "I'm trying to log in and it's been authenticating for the past, like, 20 minutes.\n"
    ciscoit_example1 = "\nAgent: ID please." \
                       "\nCaller: Sure, 234521." \
                       "\nAgent: Thanks." \
                       "\nAgent: Using an Android device?" \
                       "\nCaller: Apple." \
                       "\nAgent: I see. And is it a private WiFi network or home network?" \
                       "\nCaller: It's my home Wifi." \
                       "\nAgent: Okay could you please close all the applications and try with mobile data?" \
                       "\nCaller: Okay, okay. So get off my WiFi." \
                       "\nAgent: Absolutely correct." \
                       "\nCaller: Okay." \
                       "\nAgent: Yes, please close every application on the mobile phone and make sure you " \
                       "restart the app store." \
                       "\nCaller: Okay, got it." \
                       "\nAgent: Okay, once you've done that, could you please try installing and setting up the " \
                       "email again?" \
                       "\nCaller: Okay, just a minute." \
                       "\nAgent: No problem." \
                       "\nCaller: It's authenticating." \
                       "\nCaller: Oh hey, it worked! Thank you so much." \
                       "\nAgent: Good, I'm glad. Is there anything else you need?" \
                       "\nCaller: No, not right now. Thanks."

    gpt.add_example(Example(walmart_prompt1, walmart_example1))
    gpt.add_example(Example(ciscoit_prompt1, ciscoit_example1))
    return gpt


def test(gpt, greeting):
    prompt1 = greeting + "Caller: Hey. What should I do if my item has been recalled?\nAgent:"
    print(gpt.get_top_reply(prompt1))
    return


def get_transcript(gpt, prompt):
    response = gpt.get_top_reply(prompt)
    return ''.join([prompt, response])


if __name__ == '__main__':
    main()

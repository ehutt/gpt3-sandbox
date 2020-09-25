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

    # set generic prompt pattern
    faq_start = f"This is an FAQ for {args.kb_name}:\n"
    transcript_start = "Here is an example of it coming up in a support call."
    inject_start = "\nTranscript:\n"
    stop_sequence = "\n\n"

    # feed examples
    examples = get_examples()
    gpt = prime_model(examples,
                      prompt_prefix=faq_start,
                      prompt_suffix=transcript_start,
                      content_prefix=inject_start,
                      content_suffix=stop_sequence)

    # load questions
    if '.json' in args.query_path:
        queries, answers = load_squad_queries(args.query_path)
    elif '.csv' in args.query_path:
        queries, answers = load_faqs(args.query_path)
    else:
        raise ValueError('Supported query file types are .json (squad) and .csv (faq).')

    #random.shuffle(queries)
    print('Loaded queries.')
    output_path = args.output_path
    i = len(os.listdir(output_path))  # start count at length of dir so as not to override existing transcripts
    for j in tqdm.tqdm(range(len(queries))):
        q = queries[j]
        a = answers[j]
        print('Q{}: {}'.format(i, q))
        i += 1
        prompt = format_prompt(faq_start, transcript_start, q, a)
        transcript = get_transcript(gpt, prompt)
        save_transcript(output_path, i, transcript)


def format_prompt(faq_start, transcript_start, question, answer):
    faq = f"Question: {question}\nAnswer: {answer}\n"
    return ''.join([faq_start, faq, transcript_start])


def get_examples():
    examples = []
    transcript_start = "Here is an example of it coming up in a support call."
    faq_start = f"This is an FAQ for Walmart:\n"
    walmart_prompt1 = format_prompt(faq_start, transcript_start,
                                    "Why did I not get an email that my items are ready for Pickup Today?",
                                    "For items available for Pickup Today, we'll email/text you when it's ready "
                                    "For orders shipping from our warehouse, we'll send you an email as soon as "
                                    "the order is ready to be picked up; whether you selected to pick up from your "
                                    "local Fed Ex Office or to pick up from a Walmart store You can also use the "
                                    "Track Your Order feature in your account to follow your order's progress"
                                    )
    walmart_example1 = "\nTranscript\nAgent: Hi, thank you for calling the Walmart Help Center, my name is Jane. " \
                       "How may I help you today?\nCaller: Hi, yeah. I didn't get any emails for pick up?" \
                       "\nAgent: Okay, I can help you with that. First, can I please get your name?" \
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
                       "\nAgent: My pleasure. Thank you for calling Walmart, you have a great day.\n\n"
    examples.append((walmart_prompt1, walmart_example1))

    faq_start = f"This is an FAQ for Cisco IT:\n"
    ciscoit_prompt1 = format_prompt(faq_start, transcript_start,
                                    "How do I set up email access on my Mobile phone?",
                                    "To access email from your mobile phone, first download the MS Outlook mobile app"
                                    "from either the Apple App store of Google Play. Using a secure network, "
                                    "log in through the app using your Cisco credentials to authenticate."
                                    )
    ciscoit_example1 = "\nTranscript\nAgent: Sorry, you're speaking with Jane how may I assist you?" \
                       "\nCaller: Hi, I'm gonna set up my email. I'm on mobile. I got a new phone and, uh, " \
                       "I'm following the stuff, but the. Um, I got to the E store part downloaded E store. " \
                       "I'm trying to log in and it's been authenticating for the past, like, 20 minutes." \
                       "\nAgent: ID please." \
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
                       "\nCaller: No, not right now. Thanks.\n\n"
    #examples.append((ciscoit_prompt1, ciscoit_example1))

    faq_start = f"This is an FAQ for Farmers Insurance:\n"
    farmers_prompt1 = format_prompt(faq_start, transcript_start,
                                    "I want to receive my billing statements online and stop receiving "
                                    "paper statements. I enrolled Paperless. Why am I still receiving "
                                    "a paper billing statement in the mail?",
                                    "To receive your billing statements online, you need to enroll your billing "
                                    "account. The ‘Go Paperless' program is for policy documents only. Once you "
                                    "login to 'your account on Farmers.com, click on 'View/Pay My Bill' to go to "
                                    "the billing account section. Across the top of the screen there is a menu bar. "
                                    "Select 'Manage My Account' in the menu bar and from the drop down list: "
                                    "select the Paperless Bill option. You may select to receive a paper bill "
                                    "or stop receiving paper bills at any time. You may also select your email "
                                    "options from this screen."
                                    )
    farmers_example1 = "\nTranscript\nAgent: Thank you for calling the Farmers Help Center. My name is Jane. How may I help you?" \
                       "\nCaller: Hey, yeah I am still getting bills and things in the mail even though I " \
                       "signed up to do paperless." \
                       "\nAgent: You may still be receiving bills in the mail because going paperless " \
                       "only applies to policy documents." \
                       "\nCaller: Okay, so I can't get paperless bills?" \
                       "\nAgent: No, you still can. To go paperless with your bills, please go to your " \
                       "account online at Farmers.com, click on Pay My Bill. In the menu bar, click on " \
                       "Manage my Account and select the Paperless Bill option from the dropdown. " \
                       "This is where you can adjust all your billing settings." \
                       "\nCaller: Okay, thanks." \
                       "\nAgent: Of course, happy to help. Do you have any other questions?" \
                       "\nCaller: No.\n\n"

    examples.append((farmers_prompt1, farmers_example1))

    faq_start = f"This is an FAQ for Farmers Insurance:\n"
    farmers_prompt2 = format_prompt(faq_start, transcript_start,
                                    "What discounts are available?",
                                    "Discounts will help you save money on the premium you pay for your policy. "
                                    "You may qualify for a multi-policy discount if you insure more than just "
                                    "your bike with the same company. Specifically for motorcycles, you may be "
                                    "able to get discounts for taking safety courses, having an airbag or "
                                    "anti-lock brakes, or for being a preferred operator."
                                    )
    farmers_example2 = "\nTranscript\n" \
                       "Agent: Thank you for calling the Farmers Help Center. My name is Jane. How may I help you?" \
                       "\nCaller: I want to know if you have any special discounts right now?" \
                       "\nAgent: Sure, I would be happy to walk you through our discount options. " \
                       "What kind of insurance are you looking at?" \
                       "\nCaller: motorcycle" \
                       "\nAgent: Okay, great. So we do offer several forms of discount for motorcycle insurance." \
                       "\nCaller: Cool." \
                       "\nAgent: First, we have special multi-policy discounts for if you insure more than one " \
                       "vehicle with us. Would this apply to you?" \
                       "\nCaller: No, I just have the one bike." \
                       "\nAgent: Okay, sure. We also offer discounts for taking safety courses, " \
                       "having an air-bag or anti-lock brakes, and for being a preferred operator. " \
                       "\nCaller: Okay, cool. Thanks" \
                       "\nAgent: You're welcome.\n\n"
    examples.append((farmers_prompt2, farmers_example2))
    return examples


def prime_model(examples, prompt_prefix=None, prompt_suffix=None, content_prefix=None, content_suffix=None):
    # Construct GPT object and show some examples
    gpt = GPT(engine="davinci",
              temperature=0.9,
              max_tokens=500,
              input_prefix=prompt_prefix,
              input_suffix=prompt_suffix,
              output_prefix=content_prefix,
              output_suffix=content_suffix,
              )
    for p, e in examples:
        gpt.add_example(Example(p, e))
    return gpt


def get_transcript(gpt, prompt):
    response = gpt.get_top_reply(prompt)
    return response


if __name__ == '__main__':
    main()

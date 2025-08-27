from vlm_basics import TextToText



if __name__ == "__main__":
    t2t = TextToText()

    act_mode = "You are a supportive bot. Whatever the user says to you, you should agree with and extend their argument"
      #act_mode = "You are a disagreement bot. Whatever the user says to you, you should disagree with and provide rationale on why you are right"

    reponse = t2t.text_to_text(system_filename="output.txt", system_prompt=act_mode)

    print(reponse)

import os
from tkinter import *
import json
from PIL import Image
import cv2
from random import randint, sample

story_dir = "data/VIST-E/story_test.json"
result_A_dir = "results/VIST-E/GPT2_large_VisCon2.json"
result_B_dir = "results/VIST-E/GPT2_res_81.2935.json"
story = json.load(open(story_dir))
resutl_A = json.load(open(result_A_dir))
resutl_B = json.load(open(result_B_dir))
img_dir = "data/VIST-E/images"
max_length = 800

story = {s["story_id"]: {"context": ' '.join([s["sent1"], s["sent2"], s["sent3"], s["sent4"]]), "img_id": s["last_img_id"]} for s in story}


def natural_sentence(s):
    return s.replace(' .', '.').replace(' !', '!').replace(' ?', '?').replace(' ,', ',').replace(' \'', '\'').strip()


def sampling(n):
    sid_A = list(resutl_A.keys())
    shuffled_qid = sample(sid_A, len(sid_A))
    sampled_list = []
    sampled_num = 0
    for sid in shuffled_qid:
        # Confirm test image exists
        if not os.path.exists(img_dir + '/' + story[sid]["img_id"] + '.jpg'):
            continue

        res_A = resutl_A[sid][0]
        try:
            res_B = resutl_B[sid][0]
        except:
            continue

        if res_A == res_B:
            continue

        sampled_list.append(sid)
        sampled_num += 1

        if sampled_num >= n:
            break

    assert len(sampled_list) == n, "{} samples to be sampled but {} are sampled!".format(n, len(sampled_list))
    return sampled_list


def load_dir(sid):
    ctx = natural_sentence(story[sid]['context'])
    A_continuation = natural_sentence(resutl_A[sid][0])
    B_continuation = natural_sentence(resutl_B[sid][0])
    img = cv2.imread(img_dir + '/' + story[sid]["img_id"] + '.jpg')

    h, w = img.shape[0], img.shape[1]
    if h > max_length:
        img = cv2.resize(img, (int(max_length * w / h), max_length))
        h, w = img.shape[0], img.shape[1]
    if w > max_length:
        img = cv2.resize(img, (max_length, int(h * max_length / w)))

    return ctx, img, A_continuation, B_continuation


if __name__ == '__main__':
    dir_path = "./data/human test/"
    test_num = 250
    item_list = sampling(test_num)
    ind = 0
    random_num = randint(0, 1)
    # Model A, Model B, Not Sure
    grammar = [0, 0, 0]
    logic = [0, 0, 0]
    relation = [0, 0, 0]

    top = Tk()
    top.title('Vision Controlled Evaluation')
    top.geometry('801x600')
    top.resizable(width=True, height=True)

    frame1 = Frame(top)
    frame1.grid(row=0, column=0, sticky='w')
    ctx_title_box = Message(frame1, text="Context: ", width=800, justify=LEFT, font=20)
    ctx_title_box.pack(side='left')

    frame2 = Frame(top)
    frame2.grid(row=1, column=0, sticky='w')
    ctx_var = StringVar()
    ctx_box = Message(frame2, textvariable=ctx_var, relief=GROOVE, width=800, font=18)
    ctx_box.pack()

    frame3 = Frame(top)
    frame3.grid(row=2, column=0, sticky='w')
    ending1_title_box = Message(frame3, text="Ending 1: ", width=800, justify=LEFT, font=20)
    ending1_title_box.pack(side='left')

    frame4 = Frame(top)
    frame4.grid(row=3, column=0, sticky='w')
    ending1_var = StringVar()
    ending1_box = Message(frame4, textvariable=ending1_var, relief=GROOVE, width=800, font=18)
    ending1_box.pack()

    frame5 = Frame(top)
    frame5.grid(row=4, column=0, sticky='w')
    ending2_title_box = Message(frame5, text="Ending 2: ", width=800, justify=LEFT, font=20)
    ending2_title_box.pack(side='left')

    frame6 = Frame(top)
    frame6.grid(row=5, column=0, sticky='w')
    ending2_var = StringVar()
    ending2_box = Message(frame6, textvariable=ending2_var, relief=GROOVE, width=800, font=18)
    ending2_box.pack()

    frame7 = Frame(top)
    frame7.grid(row=6, column=0, sticky='w')
    gram_title_box = Message(frame7, text="Grammar Correctness (which one is better?): ", width=800, justify=LEFT, font=20)
    gram_title_box.pack(side='left')

    frame8 = Frame(top)
    frame8.grid(row=7, column=0, sticky='w')
    gram_var = IntVar()
    gram_but1 = Radiobutton(frame8, text="Ending 1", variable=gram_var, value=0, font=18)
    gram_but1.pack(side='left')
    gram_but2 = Radiobutton(frame8, text="Ending 2", variable=gram_var, value=1, font=18)
    gram_but2.pack(side='left')
    gram_but3 = Radiobutton(frame8, text="Not sure", variable=gram_var, value=2, font=18)
    gram_but3.pack(side='left')

    frame9 = Frame(top)
    frame9.grid(row=8, column=0, sticky='w')
    logic_title_box = Message(frame9, text="Contextual Logic (which one is better?): ", width=800, justify=LEFT, font=20)
    logic_title_box.pack(side='left')

    frame10 = Frame(top)
    frame10.grid(row=9, column=0, sticky='w')
    logic_var = IntVar()
    logic_but1 = Radiobutton(frame10, text="Ending 1", variable=logic_var, value=0, font=18)
    logic_but1.pack(side='left')
    logic_but2 = Radiobutton(frame10, text="Ending 2", variable=logic_var, value=1, font=18)
    logic_but2.pack(side='left')
    logic_but3 = Radiobutton(frame10, text="Not sure", variable=logic_var, value=2, font=18)
    logic_but3.pack(side='left')

    frame11 = Frame(top)
    frame11.grid(row=10, column=0, sticky='w')
    rela_title_box = Message(frame11, text="Visual Relation (which one is better?): ", width=800, justify=LEFT, font=20)
    rela_title_box.pack(side='left')

    frame12 = Frame(top)
    frame12.grid(row=11, column=0, sticky='w')
    rela_var = IntVar()
    rela_but1 = Radiobutton(frame12, text="Ending 1", variable=rela_var, value=0, font=18)
    rela_but1.pack(side='left')
    rela_but2 = Radiobutton(frame12, text="Ending 2", variable=rela_var, value=1, font=18)
    rela_but2.pack(side='left')
    rela_but3 = Radiobutton(frame12, text="Not sure", variable=rela_var, value=2, font=18)
    rela_but3.pack(side='left')

    def set_message(context, ending1, ending2):
        ctx_var.set(context)
        ending1_var.set(ending1)
        ending2_var.set(ending2)
        comp_var.set("Completed {}/{}".format(ind, test_num))

    def submit():
        global ind, random_num
        # Update Counter
        grammar_value = gram_var.get()
        logic_value = logic_var.get()
        relation_value = rela_var.get()
        if random_num == 0 and ind < test_num:
            grammar[grammar_value] += 1
            logic[logic_value] += 1
            relation[relation_value] += 1
        elif ind < test_num:
            if grammar_value == 2:
                grammar[2] += 1
            else:
                grammar[1-grammar_value] += 1
            if logic_value == 2:
                logic[2] += 1
            else:
                logic[1-logic_value] += 1
            if relation_value == 2:
                relation[2] += 1
            else:
                relation[1-relation_value] += 1

        # Update Boxes
        ind += 1
        if ind >= test_num:
            s = 'Thanks for completing the evaluation!'
            set_message(s, s, s)
            if ind == test_num:
                json_output = {'grammar': grammar, 'logic': logic, 'relation': relation,
                               'grammar_percent': [grammar[0] / test_num, grammar[1] / test_num],
                               'logic_percent': [logic[0] / test_num, logic[1] / test_num],
                               'relation_percent': [relation[0] / test_num, relation[1] / test_num]}
                json_output = json.dumps(json_output)
                with open('./human_results.json', 'w') as f:
                    f.write(json_output)
        else:
            random_num = randint(0, 1)
            ctx, img, A_continuation, B_continuation = load_dir(item_list[ind])
            ending1 = A_continuation if random_num == 0 else B_continuation
            ending2 = B_continuation if random_num == 0 else A_continuation
            set_message(ctx, ending1, ending2)
            cv2.imshow('Image', img)

    frame13 = Frame(top)
    frame13.grid(row=12, column=0, sticky='w', padx=100)
    submit_but = Button(frame13, text='submit', font=20, command=submit)
    submit_but.pack()

    frame14 = Frame(top)
    frame14.grid(row=13, column=0, sticky='w', padx=50)
    comp_var = StringVar()
    comp_box = Message(frame14, textvariable=comp_var, width=800, font=18)
    comp_box.pack()

    ctx, img, A_continuation, B_continuation = load_dir(item_list[ind])
    ending1 = A_continuation if random_num == 0 else B_continuation
    ending2 = B_continuation if random_num == 0 else A_continuation
    set_message(ctx, ending1, ending2)
    cv2.imshow('Image', img)

    top.mainloop()

# coding: utf-8
import json
import random
from change import change
from PIL import Image
from PIL import ImageFilter
from model.capthafactory import CaptchaFactory
from model.utils import CaptchaUtils


def char_custom_fn(single_char):
    single_char = change(single_char)
    # 字符翻转处理
    index = random.randint(0,3)
    if index == 0:
        single_char = single_char.transpose(Image.FLIP_LEFT_RIGHT)
    elif index == 1:
        single_char = single_char.transpose(Image.FLIP_TOP_BOTTOM)
    elif index == 2:
        single_char = single_char.transpose(Image.FLIP_LEFT_RIGHT)
        single_char = single_char.transpose(Image.FLIP_TOP_BOTTOM)
    # return single_char.filter(ImageFilter.GaussianBlur)
    return single_char


def bg_custom_fn(bg):
    # do something you wanted
    # return bg.filter(ImageFilter.GaussianBlur)
    return bg


def main():
    project_name = "demo"
    with open("configs/%s.json" % project_name, encoding="utf-8") as fp:
        demo_config = json.load(fp)

    demo_factory = CaptchaFactory(char_custom_fns=[char_custom_fn], bg_custom_fns=[bg_custom_fn], **demo_config)
    index =20
    while index:
        captcha = demo_factory.generate_captcha()
        captcha.save("output/%s/%s.jpg" % (project_name, captcha.text))
        # print(captcha.text, captcha.num)

        index -= 1


if __name__ == "__main__":
    main()

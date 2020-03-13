import re

def image_uid_from_fp(f):
    image_id = re.match('.*(.*/.*/.*\\.jpg)', f)
    if image_id:
        return image_id.group(1)
    else:  # not detected; invalid path
        print("[WARNING] invalid vgg identifier: {}".format(fp))
        return None
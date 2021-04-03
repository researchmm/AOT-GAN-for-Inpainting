import cv2
import os
import importlib
import numpy as np
from glob import glob 

import torch
from torchvision.transforms import ToTensor

from utils.option import args
from utils.painter import Sketcher



def postprocess(image):
    image = torch.clamp(image, -1., 1.)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return image



def demo(args):
    # load images 
    img_list = []
    for ext in ['*.jpg', '*.png']: 
        img_list.extend(glob(os.path.join(args.dir_image, ext)))
    img_list.sort()

    # Model and version
    net = importlib.import_module('model.'+args.model)
    model = net.InpaintGenerator(args)
    model.load_state_dict(torch.load(args.pre_train, map_location='cpu'))
    model.eval()

    for fn in img_list:
        filename = os.path.basename(fn).split('.')[0]
        orig_img = cv2.resize(cv2.imread(fn, cv2.IMREAD_COLOR), (512, 512))
        img_tensor = (ToTensor()(orig_img) * 2.0 - 1.0).unsqueeze(0)
        h, w, c = orig_img.shape
        mask = np.zeros([h, w, 1], np.uint8)
        image_copy = orig_img.copy()
        sketch = Sketcher(
            'input', [image_copy, mask], lambda: ((255, 255, 255), (255, 255, 255)), args.thick, args.painter)

        while True:
            ch = cv2.waitKey()
            if ch == 27:
                print("quit!")
                break

            # inpaint by deep model
            elif ch == ord(' '):
                print('[**] inpainting ... ')
                with torch.no_grad():
                    mask_tensor = (ToTensor()(mask)).unsqueeze(0)
                    masked_tensor = (img_tensor * (1 - mask_tensor).float()) + mask_tensor
                    pred_tensor = model(masked_tensor, mask_tensor)
                    comp_tensor = (pred_tensor * mask_tensor + img_tensor * (1 - mask_tensor))

                    pred_np = postprocess(pred_tensor[0])
                    masked_np = postprocess(masked_tensor[0])
                    comp_np = postprocess(comp_tensor[0])

                    cv2.imshow('pred_images', comp_np)
                    print('inpainting finish!')
            
            # reset mask
            elif ch == ord('r'):
                img_tensor = (ToTensor()(orig_img) * 2.0 - 1.0).unsqueeze(0)
                image_copy[:] = orig_img.copy()
                mask[:] = 0
                sketch.show()
                print("[**] reset!")

            # next case
            elif ch == ord('n'):
                print('[**] move to next image')
                cv2.destroyAllWindows()
                break

            elif ch == ord('k'): 
                print('[**] apply existing processing to images, and keep editing!')
                img_tensor = comp_tensor
                image_copy[:] = comp_np.copy()
                mask[:] = 0
                sketch.show()
                print("reset!")
            
            elif ch == ord('+'): 
                sketch.large_thick()

            elif ch == ord('-'): 
                sketch.small_thick()
            
            # save results
            if ch == ord('s'):
                cv2.imwrite(os.path.join(args.outputs, f'{filename}_masked.png'), masked_np)
                cv2.imwrite(os.path.join(args.outputs, f'{filename}_pred.png'), pred_np)
                cv2.imwrite(os.path.join(args.outputs, f'{filename}_comp.png'), comp_np)
                cv2.imwrite(os.path.join(args.outputs, f'{filename}_mask.png'), mask)

                print('[**] save successfully!')
        cv2.destroyAllWindows()

        if ch == 27:
            break


if __name__ == '__main__':
    demo(args)

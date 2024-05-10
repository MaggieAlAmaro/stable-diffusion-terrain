import gradio as gr
from PIL import Image
import numpy as np
import PIL
from appTerrainMesh import TexturedGLTFFromHeightmap, returnSeperateRGBA

import os, time

import scripts.sdtxt2img as txt2img
import scripts.segmentation as semseg



def loadTxt2ImgModel():
    modelParameters = [
        # "--ddim_steps", "100",
        # "--n_iter", "2",
        "--mode", "RGBA",
        # "--outdir", "outputs/terrain",
        # "--n_samples", "5",
        # "--plms",
        "--ckpt", "F:\\More Models\\2024-03-31T16-18-59_text2img-terrain-ldm-kl-8\\checkpoints\\epoch=000047.ckpt", 
        "--config", "F:\\More Models\\2024-03-31T16-18-59_text2img-terrain-ldm-kl-8\\configs\\2024-04-01T12-03-47-project.yaml"
    ]


    opt = txt2img.parseMyArgs(modelParameters)
    modelTxt, samplerTxt = txt2img.load_model_and_sampler(opt)
    return modelTxt, samplerTxt
    # progress.
    

def loadSemanticSynthModel():
    modelParameters = [
        "--plms",
        # "--ckpt", "C:\\Users\\maggi\\Desktop\\epoch=000066.ckpt",
        # "--config", "C:\\Users\\maggi\\Desktop\\2024-04-23T01-31-12-project.yaml",
        # "--ckpt", "C:\\Users\\maggi\\Desktop\\epoch=000066.ckpt",
        # "--config", "C:\\Users\\maggi\\Desktop\\2024-04-23T01-31-12-project.yaml",
        "--ckpt", "logs\\2024-03-08T21-14-52_terrain-segmentation-ldm-kl-4-going-well\\checkpoints\\epoch=000052.ckpt",
        "--config", "logs\\2024-03-08T21-14-52_terrain-segmentation-ldm-kl-4-going-well\\configs\\2024-03-11T15-01-35-project.yaml",
              
    ]
    opt = semseg.parseArguments(modelParameters)
    modelSegSem, SamplerSegSem = semseg.load_model_and_sampler(opt)
    return modelSegSem, SamplerSegSem
    


# modelTxt, samplerTxt = loadTxt2ImgModel()
# modelSegSem, SamplerSegSem = loadSemanticSynthModel()


def txt2imgmodelPayload(prompt, steps, iter, batch,seed, sampleType):
    modelParameters = [
        # "--prompt", "plain, hills, lake", 
        # "--ddim_steps", "100",
        # "--n_iter", "2",
        "--mode", "RGBA",
        # "--outdir", "outputs/terrain",
        # "--n_samples", "5",
        # "--plms",
        "--config", "F:\\More Models\\2024-03-31T16-18-59_text2img-terrain-ldm-kl-8\\configs\\2024-04-01T12-03-47-project.yaml"
    ]
    payload = modelParameters
    payload.append("--prompt")
    payload.append(str(prompt))
    payload.append("--ddim_steps")
    payload.append(str(steps))
    payload.append("--n_samples")
    payload.append(str(batch))
    payload.append("--n_iter")
    payload.append(str(iter))
    payload.append("--seed")
    payload.append(str(seed))
    payload.append(("--"+sampleType))
    print("AAWDWA",payload)
    opt = txt2img.parseMyArgs(payload)
    # sample_path, grid_path = txt2img.sampleLoop(opt, modelTxt, modelTxt)
    # sample_path, grid_path = main(payload)
    # sample_path, grid_path = main(payload)
    # return [os.path.join(sample_path,img) for img in os.listdir(sample_path)]
    return []
    #rgbaImagePath = sampling.sample(payload)
    # gltfFile = interpretInputImage(rgbaImagePath)
    # return gltfFile

import cv2
def semsegmodelPayload(filepath, steps, iter, batch, seed, sampleType):
    # img = Image.open(filepath['composite'])
    img = filepath['composite']
    img = img.resize((256, 256), Image.Resampling.NEAREST)
    data = np.array(img)

    # data = data.reshape(-1, 3)
    colorValues, counts = np.unique(data.reshape(-1, 3), 
                        return_counts = True, 
                        axis = 0)
    print(len(colorValues))
    print(colorValues)
    purpledata = np.where(data == [0,0,139], [128, 51, 128], data)
    purpledata = purpledata.astype(np.uint8)

    newImg = Image.fromarray(purpledata)

    # img.thumbnail((256,256), Image.Resampling.LANCZOS)
    drawingOut = os.path.join('outputs', "drawings", time.strftime("%Y-%m-%d"))
    os.makedirs(drawingOut, exist_ok=True)
    count_name = len(os.listdir(drawingOut))
    drawingOut = os.path.join(drawingOut, str(count_name)+".png")
    # img = Image.fromarray()
    newImg.save(drawingOut)

    modelParameters = [
        # "--n_iter", "2",
        "--f", "4",
        # "--C", "5",
        "--config", "logs\\2024-03-08T21-14-52_terrain-segmentation-ldm-kl-4-going-well\\configs\\2024-03-11T15-01-35-project.yaml",
                
        # "--config", "C:\\Users\\maggi\\Desktop\\2024-04-23T01-31-12-project.yaml",
        # "--config", "F:\\More Models\\2024-03-31T16-18-59_text2img-terrain-ldm-kl-8\\configs\\2024-04-01T12-03-47-project.yaml"
    ]

    
    # "--from-folder", "logs\\2024-03-08T21-14-52_terrain-segmentation-ldm-kl-4-going-well\\genMask",
    # "--seed", "9999",
    payload = modelParameters
    payload.append("--from-filename")
    payload.append(drawingOut)
    payload.append("--ddim_steps")
    payload.append(str(steps))
    payload.append("--n_samples")
    payload.append(str(batch))
    payload.append("--n_iter")
    payload.append(str(iter))
    payload.append("--seed")
    payload.append(str(seed))
    payload.append(("--"+sampleType))
    print("Payload: ", payload)
    opt = semseg.parseArguments(payload)
    sample_path, grid_path = semseg.sampleLoop(opt, modelSegSem, SamplerSegSem)
    return [os.path.join(sample_path,img) for img in os.listdir(sample_path)]



def interpretInputImage(inputFilename):
    print("asdw", inputFilename)
    rgbFilename, aFilename = returnSeperateRGBA(inputFilename)
    meshOut = os.path.join('outputs', "3D-models", time.strftime("%Y-%m-%d %H-%M-%S"))
    os.makedirs(meshOut,exist_ok=True)
    renderName = os.path.basename(rgbFilename).split('-')[0]
    fullRenderName = os.path.join(meshOut, renderName+ '.gltf')
    # renderCount = len(os.listdir(os.dirpath(rgbFilename)))
    TexturedGLTFFromHeightmap(aFilename,fullRenderName, rgbFilename)
    return fullRenderName, rgbFilename, aFilename
    # img = Image.open(inputFilename)
    # print(img.mode)
    # # print(inputFilename)
    # if img.mode == "RGB":
    #     img = img.convert('L')
    # elif img.mode == "RGBA":
    #     rgb, a = returnSeperateRGBA(inputFilename)
    #     TexturedGLTFFromHeightmap(a,'out-test.gltf',rgb)
    #     return os.path.join(os.path.dirname(__file__),'outputs','out-test.gltf')
    # if img.mode != "L":
    #     raise Exception("Unkown Heightmap format")
    # TexturedGLTFFromHeightmap(inputFilename,'out-test.gltf')
    

    # return os.path.join(os.path.dirname(__file__),'outputs','out-test.gltf')


def rgaTexture(inputFilename):
    rgbFilename, aFilename = returnSeperateRGBA(inputFilename)
    meshOut = os.path.join('outputs', "3D-models", time.strftime("%Y-%m-%d %H-%M-%S"))
    os.makedirs(meshOut,exist_ok=True)
    renderName = os.path.basename(rgbFilename).split('-')[0]
    fullRenderName = os.path.join(meshOut, renderName+ '.gltf')
    # renderCount = len(os.listdir(os.dirpath(rgbFilename)))
    TexturedGLTFFromHeightmap(aFilename,fullRenderName, rgbFilename)
    return fullRenderName, rgbFilename, aFilename



def draw_mask(input):
    return input["layers"][0]

def getSelectedGalleryImage(evt: gr.SelectData):
    # print(data[evt.index][0]) 
    return evt.value['image']['path']

segmentationNumber = 0
def enableNextSemDraw(seg):
    # Doesnt work
    # return gr.update(brush=myWaterBrush)
    # global segmentationNumber
    # segmentationNumber = max(0, min(segmentationNumber + 1, 2))
    return gr.update(interactive=True), seg
 

def enablePreviousSemDraw(seg):
    # Doesnt work
    # return gr.update(brush=myWaterBrush)
    # global segmentationNumber
    # segmentationNumber = max(0, min(segmentationNumber - 1, 2))
    return gr.update(interactive=True), seg

def enableNextSemDraw2(seg):
    return gr.update(interactive=True), seg

def enablePreviousSemDraw2(seg):
    return gr.update(interactive=True), seg

img =  np.full((512,512,3),255)
segmentBackground = {
    "background": img,
    "layers": [img],
    "composite":None,
    }
segmentBackground2 = {
                    "background": img,
                    "layers": [img],
                    "composite":None,
                    }
segmentBackground3 = {
                    "background": img,
                    "layers": [img],
                    "composite":None,
                    }


with gr.Blocks() as demo:
    modelVis = gr.Model3D(
        clear_color=[0.0, 0.0, 0.0, 0.0],
        interactive=False
        # transforms=False
        # crop_size="1:1",
    )

    # examples = gr.Examples(
    #     examples=[
    #         [os.path.join(os.path.dirname(__file__), "HUGEBALLS.gltf")],
    #     	],
    #     inputs=[modelVis],
        
    #     )

    with gr.Row():
        with gr.Column(): 
            texture = gr.Image(sources=[],width="20vw",)
        with gr.Column(): 
            heightmap = gr.Image(sources=[],width="20vw",)
    


    with gr.Row():
        ################# Gallery
        with gr.Column(scale=2): 
            gallery = gr.Gallery(
                label="Generated images", show_label=False, elem_id="gallery"
            , columns=[3], object_fit="contain", height="auto")   
            selectedImg = gr.Textbox(visible=False)
            gallery.select(fn=getSelectedGalleryImage,outputs=selectedImg)
            renderBtn = gr.Button("Render")
            renderBtn.click(fn=rgaTexture, inputs=selectedImg, outputs=[modelVis,texture,heightmap])

        ################# Model Parameters
        with gr.Column(scale=1):
            with gr.Group():
                    with gr.Row(): 
                        seed = gr.Number(label="Seed",value=0)
                    with gr.Row(): 
                        steps = gr.Slider(label="Steps", value=50, minimum=20,maximum=200,step=1)
                    with gr.Row(): 
                        iter = gr.Slider(label="Iterations", value=1, minimum=0,maximum=50,step=1)
                    with gr.Row(): 
                        batchN = gr.Slider(label="Batch", value=4, minimum=1,maximum=10,step=1)
                    with gr.Row(): 
                        sampleAlgorithm = gr.Radio(label="Sampling Algorithm", choices=["plms","ddim"], value="plms")
                    
        ################# Generation Tabs
        with gr.Column(): 
            with gr.Tab("Semantic Synthesis") as semTab:
                # semTab.select(fn=loadSemanticSynthModel)     
                mylandBrush = gr.Brush(
                    colors=['#ffff00','#ff8000','#808080','#ffffff' ],
                    color_mode='fixed'
                )                    
                myWaterBrush = gr.Brush(
                    colors=['#00008B','#3333ff','#00ffff',], # purple #803380
                    color_mode='fixed'
                )
                myfinalBrush = gr.Brush(
                    colors=['#00ff00','#ff0000'],
                    color_mode='fixed'
                )
                
                gr.Label("Draw Ground")
                segmentation = gr.ImageEditor(
                    value= segmentBackground,
                    sources=[],
                    type="pil",
                    transforms=[],
                    width=512,
                    height=512,
                    image_mode='RGB',
                    brush=mylandBrush,
                )         
                        
                prev_button = gr.Button("To Previous")
                next_button = gr.Button("To Next")

                gr.Label("Draw Water")
                segmentation2 = gr.ImageEditor(
                    value= segmentBackground2,
                    sources=[],
                    type="pil",
                    transforms=[],
                    width=512,
                    height=512,
                    image_mode='RGB',
                    brush=myfinalBrush,
                    visible=True
                )
                
                prev2_button = gr.Button("To Previous")
                next2_button = gr.Button("To Next")

                gr.Label("Draw Final Details")
                segmentation3 = gr.ImageEditor(
                    value= segmentBackground3,
                    sources=[],
                    type="pil",
                    transforms=[],
                    width=512,
                    height=512,
                    image_mode='RGB',
                    brush=myWaterBrush,
                    visible=True
                )
                
                exampleSegmentation = gr.Examples(
                    examples=[
                        [os.path.join(os.path.dirname(__file__), "assets","examples","mountain_4529_6589_0_768.png")],
                        [os.path.join(os.path.dirname(__file__), "assets","examples","coast_6831_8902_256_256.png")],
                        ],
                    inputs=[segmentation3],
                    
                    )

                def fillBackground():
                    return segmentBackground
                

                segmentation.clear(fn=fillBackground, outputs=segmentation)
                segmentation2.clear(fn=fillBackground, outputs=segmentation2)
                segmentation3.clear(fn=fillBackground, outputs=segmentation3)
                # colors = gr.ColorPicker(value="#ff0000",label="Red", interactive=True)
                # colors.focus(fn=changeBrush, inputs=colors)

                seg_button = gr.Button("Generate")
                next_button.click(fn=enableNextSemDraw,inputs=segmentation, outputs=[segmentation2,segmentation2])
                prev_button.click(fn=enablePreviousSemDraw,inputs=segmentation2, outputs=[segmentation,segmentation])
                next2_button.click(fn=enableNextSemDraw2,inputs=segmentation2, outputs=[segmentation3,segmentation3])
                prev2_button.click(fn=enablePreviousSemDraw2,inputs=segmentation3, outputs=[segmentation2,segmentation2])

                    # IF statements dont work
                    # if segmentationNumber == 0:
                    #     next_button.click(fn=enableNextSemDraw,inputs=segmentation, outputs=[segmentation,segmentation2,segmentation2,segmentation2])
                    # elif segmentationNumber == 1:
                    #     next_button.click(fn=enableNextSemDraw,inputs=segmentation2, outputs=[segmentation2,segmentation3,segmentation3,segmentation3])
                
                    # if segmentationNumber == 1:
                    #     prev_button.click(fn=enablePreviousSemDraw,inputs=segmentation2, outputs=[segmentation2,segmentation,segmentation,segmentation])
                    # elif segmentationNumber == 2:
                    #     prev_button.click(fn=enablePreviousSemDraw,inputs=segmentation3, outputs=[segmentation3,segmentation2,segmentation2,segmentation2])
                
            
                
                seg_button.click(fn=semsegmodelPayload, inputs=[segmentation3,steps,iter,batchN,seed,sampleAlgorithm], outputs=gallery)








            with gr.Tab("Text Generation") as txtTab:         
                text_input = gr.Textbox()
                text_button = gr.Button("Generate")
                text_button.click(fn=txt2imgmodelPayload, inputs=[text_input,steps,iter,batchN,seed,sampleAlgorithm], outputs=gallery)
            # txtTab.select(fn=loadTxt2ImgModel)       







            

            with gr.Tab("Import Heightmap"):
                gr.Markdown(
                """
                ## Supports:
                - Combination color + Heightmap 4 channel image (color RGB + heightmap A)

                - 8 bit color depth

                
                ### TODO
                - 1 Channel Heightmap 
                - 3 Channel Heightmap 
                """)
                #imgPath = gr.Image(type='filepath',image_mode='RGBA',sources=["upload","clipboard"])
                imgPath = gr.File(type='filepath',file_types=['image'])
                input_button = gr.Button("Import")
                input_button.click(fn=interpretInputImage, inputs=imgPath, outputs=[modelVis,texture,heightmap], api_name="LOLOLOL")
                
                # @seg_button.click(inputs=name, outputs=output)
                # def greet(name):
                #     return "Hello " + name + "!"
        

    
demo.launch()#share=True
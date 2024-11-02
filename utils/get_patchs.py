import pyvips

def convert_png_to_deepzoom(input_png, output_prefix, tile_size=256, overlap=0):
    # PNG 이미지를 불러옵니다.
    image = pyvips.Image.new_from_file(input_png, access="sequential")
    
    # DeepZoom 형식으로 저장합니다. quality 인수는 제거합니다.
    dz = image.dzsave(output_prefix, tile_size=tile_size, overlap=overlap, suffix='.jpg')
    
if __name__ == '__main__':
    # 입력 PNG 파일 경로와 출력 DeepZoom 파일의 접두어를 지정합니다.
    input_png_path = "/workspace/whole_slide_image_LLM/data/train_imgs/BC_01_0001.png"
    output_dz_prefix = "/workspace/whole_slide_image_LLM/data/train_svs/"

    convert_png_to_deepzoom(input_png_path, output_dz_prefix)
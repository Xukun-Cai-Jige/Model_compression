class Model:
    def __init__(self, quantization_mode):
        pass

    def process(self, texts, images, videos):
        pass

    def generate(self):
        pass

    def get_processor(self):
        pass

    def process_generate(self):
        pass

    def get_model_name(self):
        pass

    def get_model_size(self):
        return self.model.get_memory_footprint()
    
    def video_inference(self, video_path, user_query, fps=1.0):
        pass
    
    def get_average_processing_time(self):
        pass
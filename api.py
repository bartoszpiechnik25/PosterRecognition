from flask import Flask, jsonify, request, abort
from flask_restful import Resource, Api
from clip.poster_clip import PosterCLIP
from PIL import Image
import requests, io, base64, torch, os

app = Flask(__name__)
api = Api(app)

model_path = os.environ.get('MODEL_PATH')
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = PosterCLIP(device=device, path=model_path) 

class PredictMovie(Resource):
    def post(self):
        args = request.json
        if not len(args):
            abort(400, "No data provided")
        result = {}
        for key in ['url', 'base64', 'path']:
            if key not in args:
                continue
            im = self.getImage(args[key], key)
            res = MODEL.predict(im)
            prediction = max(res, key=res.get).replace('Poster of a movie: ',"")
            titles = [title.replace('Poster of a movie: ',"") for title in res.keys()]
            titles.remove(prediction)
            result[key] = {"prediction": prediction, "similar": titles}
        return jsonify(result)

    @staticmethod
    def getImage(value: str, type: str) -> Image.Image:
        if type == 'url':
            response = requests.get(value)
            if response.status_code == 200:
                return Image.open(io.BytesIO(response.content)).convert('RGB')
            raise RuntimeError("Couldn't download image")   
        if type == 'base64':
            decoded = base64.b64decode(value)
            return Image.open(io.BytesIO(decoded)).convert('RGB')
        if type == 'path':
            return Image.open(value).convert('RGB')
        raise RuntimeError("Invalid type")    

api.add_resource(PredictMovie, '/predict')

if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=5000, debug=True)
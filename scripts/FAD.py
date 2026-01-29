# Lo primero hay que intalar la librería para utilizar Frechet Audio Distance (FAD)
'pip install frechet_audio_distance'

from frechet_audio_distance import FrechetAudioDistance

frechet = FrechetAudioDistance(
    model_name="vggish",
    use_pca=False, 
    use_activation=False,
    verbose=False
)


# Specify the paths to your saved embeddings
background_embds_path = "./fad_cache_reales.npy"
eval_embds_path = "./fad_cache_generados.npy"

# Compute FAD score while reusing the saved embeddings (or saving new ones if paths are provided and embeddings don't exist yet)
fad_score = frechet.score(
    "./Datasets/FAD_originales",
    "./Datasets/FAD_predicciones",
    background_embds_path=background_embds_path,
    eval_embds_path=eval_embds_path,
    dtype="float32"
)

print(f"RESULTADO FAD SCORE: {fad_score:.4f}")
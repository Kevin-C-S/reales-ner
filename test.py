
import reales_ner.ner as ner
import reales_ner.impro as predict


ner.ner_from_str("El Amazonas está cerca del punto de inflexión de convertirse en una sabana, sugiere un estudio. La selva del Amazonas podría estar acercándose a un punto de inflexión crítico que podría hacer que este ecosistema biológicamente rico y diverso se transforme en una sabana de hierba.El destino de la selva tropical es crucial para la salud del planeta, ya que alberga una variedad única de vida animal y vegetal, almacena una enorme cantidad de carbono e influye en gran medida en los patrones climáticos globales.", "JASON.json")

# Video 1 and folder results are just place holders, it should be replaced by the user. 
predict.detect_objects_in_video('../video1.mpg', './results')

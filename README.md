# Projet-deep-learning-2022
## Branch Main
  Ceci est le rendu de notre projet

# Comment tester notre code
## `Veuillez lancer tous nos notebook sur Collab`
    Nous importerons tous les datasets directement sur collab

## Partie pré traitement d'image
  Les fichiers concernés sont `Preprocessing.ipynb` et `dataset_processed_v2`.

  <li>Notre programme va utiliser le dataset fourni pour exporter de nouvelles images traitées.</li>

## Partie traitement de données

  Les fichiers concernés sont `make_data.ipynb`, `data_augmentation` et tous les `.csv`.
  <li>Nous utilisons les images traitées et <b>data_especes</b> contenant toutes les caractéristiques de chaque classe de plante.</li>
  <li>Nous créons des <b>Dataframe</b> regroupant les chemins des images, les classes, et les caractéristiques de chaque classe</li>
  <li>Puis nous exportons les Dataframe en format <b>csv</b> dans les différents data_test_labeled...
  <li>Finalement, on utilisera ces <b>csv</b> dans nos différents programmes pour importer les images.</li>

## Image Augmentation

Pour corriger notre Dataset nous avons eu recours au `Data Augmentation`. Les fichiers concernés sont `data_augmentation.ipynb` et les `csv_v3`.
<li>Notre programme a été lancé dans chaques sous dossier du dataset_v2 pour créer de nouvelles images modifiées subtilement.</li>
<li>Après avoir utilisé cette fonction pour chaque dossier de chaque classe, nous avons réexecuter <b>make_data</b> pour créer notre dataset final : <b>dataset_processed_v3</b>.</br>
</br>
et nos <b>csv</b> finaux : <b>data_test_labeled_v3.csv</b>

    Les partie précédentes ne sont pas à executer pour tester nos modèles. Elles sont utiles si vous voulez comprendre nous avons traité nos images et créés nos données d'entrainement et de test.
    
    Comme dit précédemment, nous importerons nos datasets dans chaque modèles de Deep learning pour ne pas refaire tous ces traitements !

## Modeles de Deep Learning

Les différents modèles sont dans les fichiers `CNN_features.ipynb`, `Transfer_learning.ipynb` et `Model_Alice/CNN_Alice.ipynb`.
</br></br>
## Il ne vous reste qu'à télécharger ces fichiers sur Collab pour tester nos modèles

     Signé, Nathan Oliveira Da Silva, Léo Bernard, Youssef AGOUSSAL, BORGET Valentin, MOUILLE Rose ING3 IA A

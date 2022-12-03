### Fichiers de code :
CNN.py (executable ) lancer dans le terminal linux)
CNN_bibli.py (bibliothèque de fonction)


### Dataset : 
La forme final que les datasets doivent avoir sont dans le fichier “dataset_CNN.pdf”.
ATTENTION : le format actuel des images pré-processées 512*512 est trop grand pour être contenu dans un fichier .csv ! Il faut vraiment essayer de réduire de beaucoup ces dimensions !
C’est pourquoi pour l’instant aucune valeur n’a été donnée dans la constante “nb_pixels_images” ligne 30 de CNN.py. 
De même, sans l’information du nouveau nombre de pixels par image, pas possible de connaitre la longueur finale de “trainin.csv” et donc pas possible de remplir la constante “ligne_training_csv” ligne 28.
Enfin, concernant CNN_bibli.py, aucune valeur n’a été donnée aux constantes “largeur_image” et “longueur_image” lignes 24-25.

Le chemin jusqu’aux dataset est donné dans CNN.py ligne 19-20. Les modifier si besoin.


### Code :
Je n’ai pas pu tester le code… Je suis passé d’un cas où le label était 1 valeur à un cas ou il est un tableau de 5 valeurs.
Essayez de print(label) de la fonction  “parse_data_from_input()” pour voir si on a bien un tab 2D avec chaque case qui est un tableau (5,) avec les 5 labels associés à 1 image.
Je pense que malheurseument ce n’est pas le cas... J’ai sans dote du me foirer dans les lignes 40-43 de CNN_bibli.py.

Pareil je pense que la loss fonction ligne 108 n’est plus adaptée au nouveau format des labels. Si je ne dis pas de bêtises, il faut plutôt utiliser “categorical_crossentropy”. A tester.

Enfin, même si je n’ai pas du tout eu le temps de me pencher dessus et qu’elles ne sont pas du tout adaptée au nouveau format des labels, j’ai laissé telles quelles les deux fonctions de prédictions “prediction_1_image” et “prediction_dataset” lignes 138 et 174 de CNN_bibli.py.
Elles sont pas mal détaillées, permettent de retrouver précisément quelle prédiction à raté, son numéro exacte, de tracer la matrice de confusion, de faire des statistiques de réussite par classe,… 
A voir ce qui est récupérable dedans ou pas. 



### Optimisation le réseau :

Possibilité de commenter la fonction d’ajout de bruit “add_noise()” ligne 57 de CNN_bibli.py si mauvaises performances.

Changer la taille du batch dans la fonction “simple_train_val_generators()” ligne 68 de CNN_bibli.py. Le batch actuel est petit.

Ajouter des groupes “Conv2D – MaxPooling – Dropout” dans la fonction “simple_train_val_generators” de CNN_bibli.py. 
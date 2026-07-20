# Tarea: Investigación y Desarrollo sobre Modelos Ensemble

Introducción

[1. Modelo de ensemble y propçosito ](https://github.com/Bootcamp-IA-MAD-P7/Tarea4-Modulo2-Veru#1-qu%C3%A9-es-un-modelo-ensemble-en-machine-learning-y-cu%C3%A1l-es-su-prop%C3%B3sitoen-que-se)

[2. Diferencia conceptual entre Bagging y Boosting](https://github.com/Bootcamp-IA-MAD-P7/Tarea4-Modulo2-Veru#1-qu%C3%A9-es-un-modelo-ensemble-en-machine-learning-y-cu%C3%A1l-es-su-prop%C3%B3sitoen-que-se)

[3. Stacking como técnica de ensemble y cómo difiere su enfoque de
Bagging y Boosting](https://github.com/Bootcamp-IA-MAD-P7/Tarea4-Modulo2-Veru#1-qu%C3%A9-es-un-modelo-ensemble-en-machine-learning-y-cu%C3%A1l-es-su-prop%C3%B3sitoen-que-se)

[4. Modelos de ensemble VS individuales, según el compromiso sesgo-varianza ](https://github.com/Bootcamp-IA-MAD-P7/Tarea4-Modulo2-Veru#1-qu%C3%A9-es-un-modelo-ensemble-en-machine-learning-y-cu%C3%A1l-es-su-prop%C3%B3sitoen-que-se)

[5. AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost, Voting Classifier](https://github.com/Bootcamp-IA-MAD-P7/Tarea4-Modulo2-Veru#1-qu%C3%A9-es-un-modelo-ensemble-en-machine-learning-y-cu%C3%A1l-es-su-prop%C3%B3sitoen-que-se)

[6. Bibliografía](https://github.com/Bootcamp-IA-MAD-P7/Tarea4-Modulo2-Veru#1-qu%C3%A9-es-un-modelo-ensemble-en-machine-learning-y-cu%C3%A1l-es-su-prop%C3%B3sitoen-que-se)


# 1. ¿Qué es un modelo ensemble en Machine Learning y cuál es su propósito?¿En que se basan para mejorar el rendimiento respecto a un modelo individual?

El aprendizaje conjunto es una técnica de machine learning que agrega dos o más aprendices (p. ej. modelos de regresión, redes neuronales) para producir mejores predicciones. En otras palabras, un modelo de conjunto combina varios modelos individuales para producir predicciones más precisas que un solo modelo.1 En ocasiones, las fuentes pueden referirse a esta técnica como aprendizaje basado en comités. El aprendizaje conjunto se basa en el principio de que un grupo de aprendices produce una mayor precisión general que un aprendiz individual.2 De hecho, las investigaciones respaldan su eficacia con modelos de machine learning y redes neuronales (CNN).

Una nota sobre la terminología: aprendiz base, modelo base y, en algunos casos, estimador base se refiere al modelo o modelos individuales utilizados en los algoritmos de conjunto. La bibliografía divide además a los aprendices de base en aprendices fuertes y aprendices débiles. Los modelos o aprendices débiles se definen como aquellos que funcionan poco mejor que las conjeturas aleatorias. Para los problemas de clasificación binaria, los clasificadores débiles son más formalmente aquellos que consiguen aproximadamente un cincuenta por ciento de precisión. Por el contrario, los modelos o aprendices fuertes logran un excelente rendimiento predictivo, que en la clasificación binaria se formaliza como una precisión igual o superior al ochenta por ciento.

Tenga en cuenta que algunas fuentes confunden el aprendiz débil y el aprendiz base, dado que los métodos de conjunto, especialmente los secuenciales, impulsan eficazmente a los aprendices débiles a aprendices fuertes.

# 2. Diferencia conceptual entre Bagging y Boosting, mencionando un algoritmo representativo de cada técnica.
## Bagging

El bagging es un método paralelo homogéneo que a veces se denomina agregación de bootstrapp. Utiliza réplicas modificadas de un conjunto de datos de entrenamiento determinado para entrenar a varios aprendices base con el mismo algoritmo de entrenamiento.12 El módulo de conjunto de Scikit-learn en Python contiene funciones para implementar el bagging, como BaggingClassifier.

Más concretamente, el bagging utiliza una técnica llamada remuestreo de bootstrap para derivar múltiples conjuntos de datos nuevos a partir de un conjunto de datos de entrenamiento inicial con el fin de entrenar múltiples aprendices base. ¿Cómo funciona? Digamos que un conjunto de datos de entrenamiento contiene n ejemplos de entrenamiento. El remuestreo de bootstrap copia n instancias de datos de ese conjunto en un nuevo conjunto de datos de submuestra, con algunas instancias iniciales apareciendo más de una vez y otras excluidas por completo. Estos son ejemplos de bootstrap. La repetición de este proceso x veces produce x iteraciones del conjunto de datos original, cada una de las cuales contiene n muestras del conjunto inicial. Cada iteración del conjunto inicial se utiliza para entrenar a un aprendiz base independiente con el mismo algoritmo de aprendizaje.13

![Diagrama que representa el bagging en el contexto del aprendizaje conjunto.](https://assets.ibm.com/is/image/ibm/ensemble-learning-bagging?fmt=png-alpha&dpr=on%2C1.25&wid=960&hei=540)


Un bosque aleatorio es una extensión de bagging que denota específicamente el uso de bagging para construir conjuntos de árboles de decisión aleatorios. Esto difiere de los árboles de decisión estándar en que estos últimos muestrean cada característica para identificar la mejor para la división. Por el contrario, los bosques aleatorios muestrean iterativamente subconjuntos aleatorios de características para crear un nodo de decisión.


# 3. ¿En qué consiste el stacking como técnica de ensemble y cómo difiere su enfoque de Bagging y Boosting?

## Stacking

El stacking, o generalización apilada, es un método paralelo heterogéneo que ejemplifica lo que se conoce como metaaprendizaje. El metaaprendizaje consiste en entrenar a un metaaprendiz a partir de los resultados de varios aprendices base. El stacking entrena específicamente a varios aprendices base a partir del mismo conjunto de datos utilizando un algoritmo de entrenamiento diferente para cada aprendiz. Cada aprendiz base hace predicciones en un conjunto de datos no visto. Estas primeras predicciones del modelo se compilan y se utilizan para entrenar un modelo final, que es el metamodelo.

Diagrama que representa el stacking en el contexto del aprendizaje conjunto.
Tenga en cuenta la importancia de utilizar un conjunto de datos distinto del utilizado para entrenar a los aprendices de base con el fin de entrenar al metaaprendiz. Utilizar el mismo conjunto de datos para entrenar a los aprendices base y al metaaprendiz puede dar lugar a un sobreajuste. Esto puede requerir excluir instancias de datos de los datos de entrenamiento del aprendiz base para que sirvan como datos de prueba, que a su vez se convierten en datos de entrenamiento para el metaaprendiz. La bibliografía suele recomendar técnicas como la validación cruzada para garantizar que estos conjuntos de datos no se solapen.

Al igual que el bagging, el módulo sklearn.ensemble de Python proporciona varias funciones para implementar técnicas de apilamiento.

## Boosting
Los algoritmos de boosting son un método de conjunto secuencial. El boosting tiene muchas variaciones, pero todas siguen el mismo procedimiento general. El boosting entrena un aprendiz en algún conjunto de datos inicial, d. El aprendiz resultante suele ser débil y clasifica erróneamente muchas muestras del conjunto de datos. Al igual que el bagging, el boosting muestrea instancias del conjunto de datos inicial para crear un nuevo conjunto de datos (d2). Sin embargo, a diferencia del bagging, el boosting prioriza las instancias de datos mal clasificadas del primer modelo o aprendiz. Un nuevo aprendiz recibe formación sobre este nuevo conjunto de datos d2. A continuación, se compila un tercer conjunto de datos (d3) a partir de d1 y d2, que prioriza las muestras mal clasificadas del segundo aprendiz y las instancias en las que d1 y d2 no están de acuerdo. El proceso se repite n veces para producir n aprendices. A continuación, el boosting combina y pondera todos los aprendices para producir predicciones finales.18

Diagrama que representa el boosting en el contexto del aprendizaje conjunto.


# 4. ¿Por qué los modelos de ensemble suelen lograr mejor generalización que los modelos individuales, desde la perspectiva del compromiso sesgo-varianza?

Equilibrio entre sesgo y varianza
El equilibrio entre el sesgo y la varianza es un problema muy conocido en el machine learning y un principio que motiva muchas de técnicas de regularización. Podemos definirlos como:

- El sesgo mide la diferencia media entre los valores pronosticados y los valores reales. A medida que aumenta el sesgo, un modelo predice con menos precisión en un conjunto de datos de entrenamiento. Un sesgo alto se refiere a un error alto en el entrenamiento. La optimización significa los intentos de reducir los sesgos.

- La varianza mide la diferencia entre las predicciones en varias realizaciones de un modelo determinado. A medida que aumenta la varianza, un modelo predice con menos precisión sobre datos no vistos. Una varianza elevada implica un alto nivel de error durante las pruebas y la validación. La generalización se refiere a los intentos de reducir la varianza.

El sesgo y la varianza representan inversamente la precisión del modelo en los datos de entrenamiento y prueba, respectivamente.5 Son dos de los tres términos que comprenden la tasa de error total de un modelo, siendo el tercero el error irreducible. Este tercer término denota el error resultante de la aleatoriedad inherente a un conjunto de datos. El error total del modelo se puede definir mediante la fórmula:6

![Fórmula de error total para el aprendizaje conjunto](https://assets.ibm.com/is/image/ibm/ensemble-learning-equation-error?fmt=png-alpha&dpr=on%2C1.25&wid=960&hei=540)


# 5. Explicación breve de AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost, Voting Classifier

● AdaBoost: pondera los errores del modelo. Es decir, al crear una nueva iteración de un conjunto de datos para entrenar al siguiente alumno, AdaBoost añade ponderaciones a las muestras mal clasificadas del alumno anterior, lo que hace que el siguiente alumno priorice esas muestras mal clasificadas.

● Gradient Boosting:  Utiliza los errores residuales al entrenar a los nuevos aprendices. En lugar de ponderar las muestras mal clasificadas, el boosting de gradiente utiliza los errores residuales de un modelo anterior para establecer predicciones objetivo para el modelo siguiente. De este modo, intenta cerrar la brecha de error dejada por un modelo.

● XGBoost: Significa “Extreme Gradient Boosting”, donde el término “Gradient Boosting” se origina en el papel Greedy Function Approximation: A Gradient Boosting Machine, de Friedman. se desarrolla con una profunda consideración en términos de optimización de sistemas y principios en el aprendizaje automático. El objetivo de esta biblioteca es empujar el extremo de los límites de cálculo de las máquinas para proporcionar una biblioteca escalable, portátil y precisa.
● LightGBM: s un marco de refuerzo de gradiente que utiliza algoritmos de aprendizaje basados en árboles. Está diseñado para ser distribuido y eficiente con las siguientes ventajas:

- Velocidad de entrenamiento más rápida y mayor eficiencia.
- Menor uso de memoria.
- Mejor precisión.
- Soporte de aprendizaje paralelo, distribuido y GPU.
- Capaz de manejar datos a gran escala.

● CatBoost:  es un algoritmo para el aumento de gradientes en los árboles de decisión. Es desarrollado por investigadores e ingenieros de Yandex, y se utiliza para la búsqueda, sistemas de recomendación, asistente personal, coches autónomos, predicción del clima y muchas otras tareas en Yandex y en otras empresas, incluyendo CERN, Cloudflare, Careem taxi. Está en código abierto y puede ser utilizado por cualquier persona.

● Voting Classifier: Un clasificador de votación es una técnica de aprendizaje conjunto utilizada en los campos de la estadística, análisis de los datos, y la ciencia de datos para mejorar el rendimiento predictivo de los modelos de aprendizaje automático. Al combinar varios clasificadores, un clasificador de votación aprovecha las fortalezas de cada modelo individual para producir una predicción más sólida y precisa. Este método es particularmente eficaz en escenarios en los que diferentes modelos pueden capturar diferentes aspectos de los datos, mejorando así el proceso general de toma de decisiones.

● Stacking Classifier: La generalización apilada consiste en apilar la salida de individuo Estimador y uso de un clasificador para calcular la predicción final. Apilamiento Permite utilizar la fuerza de cada estimador individual mediante el uso de su Salida como entrada de un estimador final.


# Bibliografía

- https://www.ibm.com/es-es/think/topics/ensemble-learning
- https://xgboost.readthedocs.io/en/stable/tutorials/model.html
- https://lightgbm.readthedocs.io/en/latest/index.html
- https://catboost.ai/
- https://es.statisticseasily.com/glosario/%C2%BFQu%C3%A9-es-el-clasificador-de-votaci%C3%B3n%3F/
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html#gallery-examples

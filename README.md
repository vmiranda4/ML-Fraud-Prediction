# Detección de transacciones fraudulentas con Machine Learning

En este proyecto se desarrolla de forma completa todo el proceso para obtener un producto final que es una **aplicación Streamlit** para interactuar fácilmente con el modelo.

Toda esta primera parte se encuentra en el archivo `Pred_Fraude.ipynb`.

En primer lugar se realizar el Análisis Exploratorio de Los Datos, donde es posible evidenciar una gran cantidad de outliers. Adicionalmente, se llevan a cabo transformaciones a las variables que pueden ser útiles al momento de implementar los modelos. En total se usaron 3 tipos de transformaciones: RobustScaler, QuantileTransformer y $\ln(x+1)$.

Puesto que para usar el modelo con nuevos datos este necesita recibirlos con las transformaciones pertinentes, es necesario guardar estos escaladores como archivos '.pkl' que luego se leen en el programa para desplegar la aplicación.

Adicional a esto, también se realizó la imputación de datos para la columna 'C', que poseía datos nulos. Después de obtener las features más significativas para explicar el target, esta hizo parte de la lista por lo que, considerando que en producción se pueden recibir más datos nulos, también fue necesario guardar el imputador como archivo '.pkl' para leerlo luego.

Finalmente, después de haber realizado el preprocesamiento adecuado de los datos, se empezaron a implementar diferentes modelos y para evaluar cuál de todos tenía mejor desempeño se optó por usar la matriz de confusión, pues lo que se quiere maximizar solo los verdaderos positivos.

En este punto del desarrollo fue necesario hacer una búsqueda exhaustiva dado que el conjunto de datos está desbalanceado (tiene una gran cantidad de datos de la clase no fraude en comparación con la clase de fraude). Esto en principio causaba que un modelo entrenado con la proporción de las clases original aprendiera muy bien a predecir lo que es no fraude puesto que tenía más observaciones de esta clase, mientras que tuviera un mal desempeño prediciendo lo que sí es fraude.

Para abordar esto, se indagó sobre diferentes técnicas de **undersampling**, que consiste en disminuir la cantidad de datos de la clase dominante para que sea más proporcional a la clase minoritaria. En este sentido, es posible resumir las técnicas usadas en esta lista:

- Undersampling aleatorio: reduce el número de las observaciones de la clase dominante de forma aleatoria a que sea la misma de la clase minoritaria.

- TomekLinks: que elimina los vecinos más cercanos de la clase minoritaria que pertenezcan a la clase dominante. De esta forma incluso puede ayudar a remover ruido del modelo. No remueve datos para que las clases queden balanceadas necesariamente.

- NearMiss

- ClusterCentroids: agrupa los datos de la clase dominante en clusters y los datos que se toman son los centroides de estos clusters, es decir, como un promedio de los datos del cluster. Tampoco deja el conjunto completamente equilibrado.

En general, la técnica que mejores resultados tuvo por sí misma fue el undersampling aleatorio. Sin embargo, se me ocurrió que sería posible mejorar la matriz de confusión (de forma general, no solo los verdaderos positivos o verdaderos negativos) mezclando diferentes técnicas para obtener un conjunto de entrenamiento más representativo. Así es como elegí combinar TomekLinks con Undersampling aleatorio y en general se obtuvo muy bueno resultados. Adicionalmente y no tan satisfecha con los resultados, decidí buscar qué otras técnicas de undersampling podría utilizar y encontré sobre **probability calibration**, que ayuda a asignar de forma correcta las probabilidades que coinciden con las frecuencias reales, por lo que mi último intento fue tomar los modelos con las mejores matrices de confusión de todo el desarrollo anterior y usar esta calibración de probabilidades con ellos.

Finalmente el modelo con el que obtuve la mejor cantidad de falsos negativos (o sea casos que sí son fraude pero clasificó como no fraude) fue con XGBoost entrenado con la data que combina TomekLinks con undersampling aleatorio y además con calibración de probabilidades tipo `isotonic`. Con este modelo se observa la data de test y también se tienen buenos resultados, lo que indica que el modelo pudo generalizar bien. Finalmente, este es el que se guarda en un archivo `.pkl` para usarlo en la app de Streamlit.

Teniendo entonces todo lo necesario para desplegar el modelo, este se construye en el archivo `streamlit_app.py`.

### Conclusiones y observaciones finales

Todo el proceso fue muy enriquecedor para mí. Desde consultar sobre diferentes técnicas y poder implementarlas, además de explorar un poco la creatividad tanto en el preprocesamiento como durante el entrenamiento del modelo, creo que fue una gran experiencia. Además de esto, fue interesante ver cómo la diferencia en la cantidad de datos de cada clase afectaba el desempeño del modelo, y todas las ideas que pueden surgir para encontrar un equilibrio, porque siempre que mejoraban los verdaderos positivos, los verdaderos negativos se veían afectados. En general, considero que los resultados son aceptables y de igual forma se me siguen ocurriendo ideas para intentar mejorar el modelo, tal vez con diferentes transformaciones, o incluso con diferentes variables.

También fue para mí muy entretenido aprender sobre cómo desplegar un programa, de forma que sea simple y accesible la posibilidad de aprovechar el modelo entrenado. De igual forma, encontré que el desarrollo de la API se me hizo muy similar al de la app, aunque no encontré una forma de conectar ambas y tuve que probar la API por aparte.

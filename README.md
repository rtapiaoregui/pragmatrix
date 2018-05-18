# pragmatrix
my DSR Project: Pragmatrix

**El evaluador de calidad literaria**

Consiste en un sistema de clasificación de texto en inglés, cuyo lenguaje es analizado a partir de sus elementos constituyentes para permitir al que lo introduzca en el sistema aventurar su procedencia. Me decidí a crear este sistema porque llevo los últimos años volcada en desentrañar lo que significa la literatura dependiendo del lenguaje en que quede de manifiesto y creo que es importante, en aras de aproximar la materia con un método científico, en primer lugar, a través de la ciencia de datos, sentar las bases de lo que la intención del emisor condiciona el estilo con el que queda retratado su mensaje. 


**El ideal que intenta alcanzar el emisor con su mensaje**

He resuelto centrarme en cifrar a código lo que supone la intención comunicativa del emisor en cómo queda plasmado el mensaje, porque creo firmemente que el que el emisor tenga la pragmática del lenguaje en el que se expresa en cuenta a la hora de articularlo le permite elegir el imaginario en que desea dar a luz a su criatura discursiva. 

El objeto de estudio de la pragmática se concibe, pues, en lingüística, como la dimensión en la que el hablante puede recalibrar la medida en que quiere que la relación entre lo que acaba diciendo y lo que declaran por su cuenta cada una de las partículas en que verbaliza su mensaje sea más o menos lineal o aditiva. La pragmática es la disciplina que estudia lo que lleva a que los estilos con los que se puede llegar a expresar un mensaje varíen entre sí; mide lo que se presta un mensaje a ser observado desde una perspectiva hermenéutica. Es, por ende, la que examina el aspecto de la expresión lingüística que más información aporta acerca del emisor y sus circunstancias. A consecuencia, he creído oportuno crear un sistema que contribuya a definir lo que analiza la pragmática. 

Idealmente, en un futuro, si se quisiera seguir aquilatando la precisión del modelo de clasificación binaria, este debería acabar siendo capaz de intuir el valor simbólico o alegórico de los textos que evalúe, o, expresado de otro modo, la medida en la que cumplen una función poética o estética. Sin embargo, para el futuro más inmediato, me he propuesto construir una herramienta para reducir el dataset a lo que lo caracteriza numéricamente como perteneciente a una de dos categorías, que paso a exponer a continuación, y que se pueda conjugar con los algoritmos que se emplean a día de hoy en el campo del procesamiento de lenguaje natural.

En suma, lo que quiero es aproximar la proporción de actos directos vs. indirectos que se dan en cada observación. A saber, calcular el valor connotativo de cada mensaje.

De referencia, los actos directos son aquellos enunciados en los que el aspecto locutivo e ilocutivo coinciden, es decir, se expresa directamente la intención. Por el contrario, los indirectos son aquellos enunciados en los que el aspecto locutivo e ilocutivo no coinciden, de lo que se infiere que la finalidad de la oración es distinta a lo que se expresa directamente.


**El dataset principal**

Las observaciones del dataset están exclusivamente formadas por texto de una extensión similar, de forma que fuesen únicamente las características inherentes de los mismos las que se prestaran a ser evaluadas, y recogido en cantidades equivalentes de diversas fuentes, que pueden clasificarse como sigue. 


Categoría uno:

Mensajes redactados con intención, en apariencia, únicamente informativa, de los que se pueda presuponer que recogen actos de habla constatativos  y oraciones que Aristóteles hubiera tenido por apofánticas.

Por ejemplo, artículos de Wikipedia y de blogs lingüísticos, así como noticias.


Categoría dos: 

Mensajes redactados con intención ilustrativa

Por ejemplo, obras de literatura, tanto relatos como novelas. 


Esta ordenación de las fuentes de procedencia de los textos es, a su vez, de la que me he valido para crear las dos clases de la variable de respuesta que he bautizado con el nombre de ‘literariedad’. 

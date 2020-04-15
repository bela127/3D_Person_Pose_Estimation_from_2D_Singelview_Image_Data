import tensorflow as tf 

def main():
    execute_eager = False
    test_nr = 6
    tf.config.experimental_run_functions_eagerly(execute_eager)
    
    images = tf.constant(100.,shape=[1,10,10,20])
    if execute_eager or test_nr == 1:
        scale = Scaler1()
        out = scale(images)
        print(out)
    if execute_eager or test_nr == 2:
        scale = Scaler2()
        out = scale(images)
        print(out)
    if execute_eager or test_nr == 3:
        scale = Scaler3()
        out = scale(images)
        print(out)
    if execute_eager or test_nr == 4:
        scale = Scaler4()
        out = scale(images)
        print(out)
    if execute_eager or test_nr == 5:
        scale = Scaler5()
        out = scale(images)
        print(out)
    if execute_eager or test_nr == 6:
        scale = Scaler6()
        out = scale(images)
        print(out)

class Scaler1(tf.keras.layers.Layer):
    
    def __init__(self, count = 4, name = "Scaler", **kwargs):
        self.count = tf.cast(count, dtype = tf.float32)
        super().__init__(name = name, **kwargs)
        self.sized_images=[]
         
    @tf.function
    def call(self, inputs):
        images = inputs
        
        image_size = tf.cast(tf.shape(images)[1:3], dtype=tf.float32)
                   
        for i in range(int(self.count)):
            scale = image_size * (1 + tf.cast(i, dtype=tf.float32))
            sized_image = tf.image.resize(images, tf.cast(scale + 0.5, dtype = tf.int32))
            self.sized_images.append(sized_image)
        
        return self.sized_images

class Scaler2(tf.keras.layers.Layer):
    
    def __init__(self, count = 5, name = "Scaler", **kwargs):
        self.count = tf.cast(count, dtype = tf.float32)
        super().__init__(name = name, **kwargs)
        self.sized_images=[]
         
    @tf.function
    def call(self, inputs):
        images = inputs
        
        image_size = tf.cast(tf.shape(images)[1:3], dtype=tf.float32)
                   
        for i in tf.range(self.count):
            scale = image_size * (1 + tf.cast(i, dtype=tf.float32))
            sized_image = tf.image.resize(images, tf.cast(scale + 0.5, dtype = tf.int32))
            self.sized_images.append(sized_image)
        
        return self.sized_images
    
class Scaler4(tf.keras.layers.Layer):
    
    def __init__(self, count = 5, name = "Scaler", **kwargs):
        self.count = tf.cast(count, dtype = tf.float32)
        super().__init__(name = name, **kwargs)
        self.sized_images=[]
         
    @tf.function
    def call(self, inputs):
        images = inputs
        
        image_size = tf.cast(tf.shape(images)[1:3], dtype=tf.float32)
        
        i = 1
        scale = image_size * (1 + tf.cast(i, dtype=tf.float32))
        sized_image1 = tf.image.resize(images, tf.cast(scale + 0.5, dtype = tf.int32))
        if i == self.count:
            return [sized_image1]
        else:
            i = 2
            scale = image_size * (1 + tf.cast(i, dtype=tf.float32))
            sized_image2 = tf.image.resize(images, tf.cast(scale + 0.5, dtype = tf.int32))
            if i == self.count:
                return [sized_image1,sized_image2]
            else:
                i = 3
                scale = image_size * (1 + tf.cast(i, dtype=tf.float32))
                sized_image3 = tf.image.resize(images, tf.cast(scale + 0.5, dtype = tf.int32))
                if i == self.count:
                    return [sized_image1,sized_image2,sized_image3]
                else:
                    i = 4
                    scale = image_size * (1 + tf.cast(i, dtype=tf.float32))
                    sized_image4 = tf.image.resize(images, tf.cast(scale + 0.5, dtype = tf.int32))
                if i == self.count:
                    return [sized_image1,sized_image2,sized_image3,sized_image4]
                else:
                    return [sized_image1,sized_image2,sized_image3,sized_image4]
    
class Scaler3(tf.keras.layers.Layer):
    
    def __init__(self, count = 5, name = "Scaler", **kwargs):
        self.count = tf.cast(count, dtype = tf.float32)
        super().__init__(name = name, **kwargs)
        self.sized_images=[]
         
    @tf.function
    def call(self, inputs):
        images = inputs
        
        image_size = tf.cast(tf.shape(images)[1:3], dtype=tf.float32)
             
        self.sized_images = [tf.image.resize(images, tf.cast(image_size * (1 + i) + 0.5, dtype = tf.int32)) for i in tf.range(self.count)]     
        
        return self.sized_images
    
class Scaler5(tf.keras.layers.Layer):
    
    def __init__(self, count = 5, name = "Scaler", **kwargs):
        self.count = tf.cast(count, dtype = tf.float32)
        super().__init__(name = name, **kwargs)
         
 
    @tf.function
    def call(self, inputs):
        images = inputs
                
        image_size = tf.cast(tf.shape(images)[1:3], dtype=tf.float32)
        
        sized_images = []
        for i in tf.range(self.count):
            scale = image_size * (1 + i)
            sized_images.append(tf.image.resize(images,tf.cast(scale + 0.5, dtype = tf.int32)))
        
        return sized_images
    
class Scaler6(tf.keras.layers.Layer):
    
    def __init__(self, count = 5, name = "Scaler", **kwargs):
        self.count = tf.cast(count, dtype = tf.float32)
        super().__init__(name = name, **kwargs)

    @tf.function
    def call(self, inputs):
        images = inputs
                
        image_size = tf.cast(tf.shape(images)[1:3], dtype=tf.float32)
        
        scales_arr = tf.TensorArray(dtype =tf.float32, size=tf.cast(self.count, dtype=tf.int32),dynamic_size=False)
        for i in tf.range(self.count):
            scale = image_size * (1 + i)
            scales_arr = scales_arr.write(tf.cast(i, dtype=tf.int32), scale)
        scales = scales_arr.stack()
        
        scales_list = tf.unstack(scales)
        sized_images = []
        for scale in scales_list:
            sized_images.append(tf.image.resize(images,tf.cast(scale + 0.5, dtype = tf.int32)))

        return sized_images
    

if __name__ == '__main__':
    main()
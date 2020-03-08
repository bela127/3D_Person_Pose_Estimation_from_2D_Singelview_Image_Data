import tensorflow as tf
import ShAReD_Net.training.loss.base as loss_base

def get_train_model():     
    return loss_base.LossTestTrainingsModel

def main():
    keypoints = 15
    x = y = z = 250
    singel_gt = tf.constant([[x+245*kp,y+204*kp,z+200*kp] for kp in range(keypoints)],dtype = tf.float32)
    dataset = tf.data.Dataset.from_tensors(([1.], singel_gt))

    mirrored_strategy = tf.distribute.MirroredStrategy()
    
    train(get_train_model, dataset, mirrored_strategy)

def train(get_train_model, dataset, dist_strat, batch_size = 8, learning_rate = 0.01):
    with dist_strat.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        train_model = get_train_model()
    
    dataset = dataset.repeat(-1).batch(batch_size)
    dist_dataset = dist_strat.experimental_distribute_dataset(dataset)
    
    def train_loop(dist_dataset, steps):
        step = tf.Variable(0, trainable=False, dtype = tf.int32)

        def train_step(inputs):
            
            def singel_device_train_step(inputs):
                with tf.GradientTape() as tape:
                    loss = train_model(inputs)
                    loss_per_input = loss / batch_size
                
                trainable_vars = train_model.trainable_vars()
                gradients = tape.gradient(loss_per_input, trainable_vars)
                [capped_gradients], _ = tf.clip_by_global_norm([gradients], 10.)
                optimizer.apply_gradients([(capped_gradients, trainable_vars)])
                
            
            per_example_losses = dist_strat.experimental_run_v2(singel_device_train_step, args=(inputs,))
            mean_loss = dist_strat.reduce(tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0)
            return mean_loss
            
        with dist_strat.scope():
            for inputs in dist_dataset:
                step += 1
                if steps > 0 and step > steps:
                    break
                loss = train_step(inputs)
                tf.print(loss)
        
    train_loop(dist_dataset, 100)
     

def save_checkpoint(model, name):
    pass

def load_checkpoint(model, name):
    pass


def validation_loop():
    pass

if __name__ == "__main__":
    main()
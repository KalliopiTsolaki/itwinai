import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import glob
import h5py
import numpy as np
from collections import defaultdict


from .base import ItwinaiBasePlModule, ItwinaiBasePlDataModule
from itwinai.models.threedgan_gen_disc_models import *


class MyDataset(Dataset):
    def __init__(self, datapath="/afs/cern.ch/work/k/ktsolaki/private/projects/GAN_scripts/3DGAN/Accelerated3DGAN/src/Accelerated3DGAN/data/*.h5"):
        self.datapath = datapath
        self.data = self.fetch_data(self.datapath)

    def __len__(self):
        return len(self.data["X"])

    def __getitem__(self, idx):
        return {"X": self.data["X"][idx], "Y": self.data["Y"][idx], "ang": self.data["ang"][idx], "ecal": self.data["ecal"][idx]}
        #self.data[idx]

    def fetch_data(self, datapath):

        print("Searching in :", datapath)
        Files = sorted(glob.glob(datapath))
        print("Found {} files. ".format(len(Files)))

        concatenated_datasets = []
        for datafile in Files:
          f=h5py.File(datafile,'r')
          dataset = self.GetDataAngleParallel(f)
          concatenated_datasets.append(dataset)
          result = {key: [] for key in concatenated_datasets[0].keys()}  # Initialize result dictionary
          for d in concatenated_datasets:
            for key in result.keys():
              result[key].extend(d[key])
        return result

    def GetDataAngleParallel(
    self,
    dataset,
    xscale=1,
    xpower=0.85,
    yscale=100,
    angscale=1,
    angtype="theta",
    thresh=1e-4,
    daxis=-1,):
      """Preprocess function for the dataset

      Args:
          dataset (str): Dataset file path
          xscale (int, optional): Value to scale the ECAL values. Defaults to 1.
          xpower (int, optional): Value to scale the ECAL values, exponentially. Defaults to 1.
          yscale (int, optional): Value to scale the energy values. Defaults to 100.
          angscale (int, optional): Value to scale the angle values. Defaults to 1.
          angtype (str, optional): Which type of angle to use. Defaults to "theta".
          thresh (_type_, optional): Maximum value for ECAL values. Defaults to 1e-4.
          daxis (int, optional): Axis to expand values. Defaults to -1.

      Returns:
        Dict: Dictionary containning the preprocessed dataset
      """
      X = np.array(dataset.get("ECAL")) * xscale
      Y = np.array(dataset.get("energy")) / yscale
      X[X < thresh] = 0
      X = X.astype(np.float32)
      Y = Y.astype(np.float32)
      ecal = np.sum(X, axis=(1, 2, 3))
      indexes = np.where(ecal > 10.0)
      X = X[indexes]
      Y = Y[indexes]
      if angtype in dataset:
          ang = np.array(dataset.get(angtype))[indexes]
      # else:
      # ang = gan.measPython(X)
      X = np.expand_dims(X, axis=daxis)
      ecal = ecal[indexes]
      ecal = np.expand_dims(ecal, axis=daxis)
      if xpower != 1.0:
          X = np.power(X, xpower)

      Y = np.array([[el] for el in Y])
      ang = np.array([[el] for el in ang])
      ecal = np.array([[el] for el in ecal])

      final_dataset = {"X": X, "Y": Y, "ang": ang, "ecal": ecal}

      return final_dataset


class MyDataModule(ItwinaiBasePlDataModule):
    def __init__(self, batch_size: int = 50, datapath="/afs/cern.ch/work/k/ktsolaki/private/projects/GAN_scripts/3DGAN/Accelerated3DGAN/src/Accelerated3DGAN/data/*.h5"):
        super().__init__()
        self.batch_size = batch_size
        self.datapath = datapath

    #def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

    def setup(self, stage: str = None):
        # make assignments here (val/train/test split)
        # called on every process in DDP

        if stage == 'fit' or stage is None:
            self.dataset = MyDataset(self.datapath) # train=True
            #print(len(self.dataset))
            #(X, Y, ang, ecal) = self.dataset[10]
            #print(Y)
            dataset_length = len(self.dataset)
            split_point = dataset_length // 10
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [split_point, dataset_length - split_point])
            #self.val_dataset = MyDataset(self.data_dir, train=False)

        #if stage == 'test' or stage is None:
            #self.test_dataset = MyDataset(self.data_dir, train=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    #def test_dataloader(self):
        #return DataLoader(self.test_dataset, batch_size=self.batch_size)


class ThreeDGAN(ItwinaiBasePlModule):
    def __init__(self, latent_size=256, batch_size=50, loss_weights=[3, 0.1, 25, 0.1], power=0.85, lr=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization=False

        self.latent_size = latent_size
        self.batch_size = batch_size
        self.loss_weights = loss_weights
        self.lr = lr
        self.power = power

        self.generator = Generator(self.hparams.latent_size)
        self.discriminator = Discriminator(self.hparams.power)

        self.epoch_gen_loss = []
        self.epoch_disc_loss = []
        self.index = 0
        self.train_history = defaultdict(list)
        self.test_history = defaultdict(list)
        #self.pklfile = "/content/drive/My Drive/Colab Notebooks/CERN/3dgan_lightning_integration/"
        self.gen_losses = []
        #self.real_batch_loss = []
        #self.fake_batch_loss = []
        #self.gen_batch_loss = []


    def BitFlip(self, x, prob=0.05):
        """
        Flips a single bit according to a certain probability.

        Args:
            x (list): list of bits to be flipped
            prob (float): probability of flipping one bit

        Returns:
            list: List of flipped bits

        """
        x = np.array(x)
        selection = np.random.uniform(0, 1, x.shape) < prob
        x[selection] = 1 * np.logical_not(x[selection])
        return x

    def mean_absolute_percentage_error(self, y_true, y_pred):
        return torch.mean(torch.abs((y_true - y_pred) / (y_true + 1e-7))) * 100

    def compute_global_loss(self, labels, predictions, loss_weights=[3, 0.1, 25, 0.1]):
        # Can be initialized outside
        binary_crossentropy_object = nn.BCEWithLogitsLoss(reduction='none')
        #there is no equivalent in pytorch for tf.keras.losses.MeanAbsolutePercentageError --> I'm using the custom "mean_absolute_percentage_error" above!
        mean_absolute_percentage_error_object1 = self.mean_absolute_percentage_error(predictions[1], labels[1])
        mean_absolute_percentage_error_object2 = self.mean_absolute_percentage_error(predictions[3], labels[3])
        mae_object = nn.L1Loss(reduction='none')

        binary_example_loss = binary_crossentropy_object(predictions[0], labels[0]) * loss_weights[0]

        #mean_example_loss_1 = mean_absolute_percentage_error_object(predictions[1], labels[1]) * loss_weights[1]
        mean_example_loss_1 = mean_absolute_percentage_error_object1 * loss_weights[1]

        mae_example_loss = mae_object(predictions[2], labels[2]) * loss_weights[2]

        #mean_example_loss_2 = mean_absolute_percentage_error_object(predictions[3], labels[3]) * loss_weights[3]
        mean_example_loss_2 = mean_absolute_percentage_error_object2 * loss_weights[3]

        binary_loss = binary_example_loss.mean()
        mean_loss_1 = mean_example_loss_1.mean()
        mae_loss = mae_example_loss.mean()
        mean_loss_2 = mean_example_loss_2.mean()

        return [binary_loss, mean_loss_1, mae_loss, mean_loss_2]

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        image_batch, energy_batch, ang_batch, ecal_batch = batch['X'], batch['Y'], batch['ang'], batch['ecal']

        image_batch = image_batch.permute(0, 4, 1, 2, 3)

        print(type(image_batch))
        print(image_batch.size())
        print(energy_batch.size())
        print(ang_batch.size())
        print(ecal_batch.size())
        #print(len(image_batch))

        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_batch = image_batch.to(self.device)
        energy_batch = energy_batch.to(self.device)
        ang_batch = ang_batch.to(self.device)
        ecal_batch = ecal_batch.to(self.device)

        #image_batch = dataset.get("X")
        #energy_batch = dataset.get("Y")
        #ecal_batch = dataset.get("ecal")
        #ang_batch = dataset.get("ang")

        optimizer_discriminator, optimizer_generator = self.optimizers()

        noise = torch.randn((self.batch_size, self.latent_size - 2)).to(self.device)
        generator_ip = torch.cat(
            (energy_batch.view(-1, 1), ang_batch.view(-1, 1), noise),
            dim=1,)
        generated_images = self.generator(generator_ip)

        # Train discriminator first on real batch
        fake_batch = self.BitFlip(np.ones(self.batch_size).astype(np.float32))
        fake_batch = torch.tensor([[el] for el in fake_batch]).to(self.device)
        labels = [fake_batch, energy_batch, ang_batch, ecal_batch]

        predictions = self.discriminator(image_batch)
        print("calculating real_batch_loss...")
        real_batch_loss = self.compute_global_loss(
            labels, predictions, self.loss_weights)
        self.log("real_batch_loss", sum(real_batch_loss), prog_bar=True)
        print("real batch disc train")
        #the following 3 lines correspond to
        #gradients = tape.gradient(real_batch_loss, discriminator.trainable_variables)
        #optimizer_discriminator.apply_gradients(zip(gradients, discriminator.trainable_variables)) in Tensorflow
        optimizer_discriminator.zero_grad()
        self.manual_backward(sum(real_batch_loss))
        #sum(real_batch_loss).backward()
        #real_batch_loss.backward()
        optimizer_discriminator.step()

        # Train discriminator on the fake batch
        fake_batch = self.BitFlip(np.zeros(self.batch_size).astype(np.float32))
        fake_batch = torch.tensor([[el] for el in fake_batch]).to(self.device)
        labels = [fake_batch, energy_batch, ang_batch, ecal_batch]

        predictions = self.discriminator(generated_images)

        fake_batch_loss = self.compute_global_loss(
            labels, predictions, self.loss_weights)
        self.log("fake_batch_loss", sum(fake_batch_loss), prog_bar=True)
        print("fake batch disc train")
        #the following 3 lines correspond to
        #gradients = tape.gradient(fake_batch_loss, discriminator.trainable_variables)
        #optimizer_discriminator.apply_gradients(zip(gradients, discriminator.trainable_variables)) in Tensorflow
        optimizer_discriminator.zero_grad()
        self.manual_backward(sum(fake_batch_loss))
        #sum(fake_batch_loss).backward()
        optimizer_discriminator.step()

        trick = np.ones(self.batch_size).astype(np.float32)
        fake_batch = torch.tensor([[el] for el in trick]).to(self.device)
        labels = [fake_batch, energy_batch.view(-1, 1), ang_batch, ecal_batch]

        gen_losses_train = []
        # Train generator twice using combined model
        for _ in range(2):
            noise = torch.randn((self.batch_size, self.latent_size - 2))
            generator_ip = torch.cat(
                (energy_batch.view(-1, 1), ang_batch.view(-1, 1), noise),
                dim=1,)

            generated_images = self.generator(generator_ip)
            predictions = self.discriminator(generated_images)
            loss = self.compute_global_loss(
                labels, predictions, self.loss_weights)
            self.log("loss", sum(loss), prog_bar=True)
            print("gen train")
            optimizer_generator.zero_grad()
            self.manual_backward(sum(loss))
            #sum(loss).backward()
            optimizer_generator.step()

            for el in loss:
                gen_losses_train.append(el)

        #generator_loss = [(a + b) / 2 for a, b in zip(*self.gen_losses)]

        '''
        # I'm not returning anything as in pl you do not return anything when you back-propagate manually
        #return_loss = real_batch_loss
        real_batch_loss = [real_batch_loss[0], real_batch_loss[1], real_batch_loss[2], real_batch_loss[3]]
        fake_batch_loss = [fake_batch_loss[0], fake_batch_loss[1], fake_batch_loss[2], fake_batch_loss[3]]
        gen_batch_loss = [gen_losses_train[0], gen_losses_train[1], gen_losses_train[2], gen_losses_train[3]]
        self.gen_losses.append(gen_batch_loss)
        gen_batch_loss = [gen_losses_train[4], gen_losses_train[5], gen_losses_train[6], gen_losses_train[7]]
        self.gen_losses.append(gen_batch_loss)

        real_batch_loss = [el.detach().numpy() for el in real_batch_loss]
        real_batch_loss_total_loss = np.sum(real_batch_loss)
        new_real_batch_loss = [real_batch_loss_total_loss]
        for i_weights in range(len(real_batch_loss)):
          new_real_batch_loss.append(real_batch_loss[i_weights] / self.loss_weights[i_weights])
        real_batch_loss = new_real_batch_loss

        fake_batch_loss = [el.detach().numpy() for el in fake_batch_loss]
        fake_batch_loss_total_loss = np.sum(fake_batch_loss)
        new_fake_batch_loss = [fake_batch_loss_total_loss]
        for i_weights in range(len(fake_batch_loss)):
          new_fake_batch_loss.append(fake_batch_loss[i_weights] / self.loss_weights[i_weights])
        fake_batch_loss = new_fake_batch_loss

        # if ecal sum has 100% loss(generating empty events) then end the training
        if fake_batch_loss[3] == 100.0 and self.index > 10:
          print("Empty image with Ecal loss equal to 100.0 for {} batch".format(self.index))
          torch.save(self.generator.state_dict(), "generator_weights.pth")
          torch.save(self.discriminator.state_dict(), "discriminator_weights.pth")
          print("real_batch_loss", real_batch_loss)
          print("fake_batch_loss", fake_batch_loss)
          sys.exit()

        # append mean of discriminator loss for real and fake events
        self.epoch_disc_loss.append([(a + b) / 2 for a, b in zip(real_batch_loss, fake_batch_loss)])

        self.gen_losses[0] = [el.numpy() for el in self.gen_losses[0]]
        gen_losses_total_loss = np.sum(self.gen_losses[0])
        new_gen_losses = [gen_losses_total_loss]
        for i_weights in range(len(self.gen_losses[0])):
          new_gen_losses.append(self.gen_losses[0][i_weights] / self.loss_weights[i_weights])
        self.gen_losses[0] = new_gen_losses

        self.gen_losses[1] = [el.detach().numpy() for el in self.gen_losses[1]]
        gen_losses_total_loss = np.sum(self.gen_losses[1])
        new_gen_losses = [gen_losses_total_loss]
        for i_weights in range(len(self.gen_losses[1])):
          new_gen_losses.append(self.gen_losses[1][i_weights] / self.loss_weights[i_weights])
        self.gen_losses[1] = new_gen_losses

        generator_loss = [(a + b) / 2 for a, b in zip(*self.gen_losses)]

        self.epoch_gen_loss.append(generator_loss)

        self.index += 1

        #logging of gen and disc loss done by Trainer
        #self.log('epoch_gen_loss', self.epoch_gen_loss, on_step=True, on_epoch=True)
        #self.log('epoch_disc_loss', self.epoch_disc_loss, on_step=True, on_epoch=True)

        # !!!have in mind the return of the loss key here (I chose arbitrarily to return the generator_loss, as I already perform back propagation above for each loss tarining)
        # "loss": return_loss[0],
        return {"epoch_disc_loss": self.epoch_disc_loss, "epoch_gen_loss": self.epoch_gen_loss}

    def training_epoch_end(self, outputs):
        discriminator_train_loss = np.mean(np.array(self.epoch_disc_loss), axis=0)
        generator_train_loss = np.mean(np.array(self.epoch_gen_loss), axis=0)

        self.train_history["generator"].append(generator_train_loss)
        self.train_history["discriminator"].append(discriminator_train_loss)

        print("-" * 65)
        ROW_FMT = ("{0:<20s} | {1:<4.2f} | {2:<10.2f} | {3:<10.2f}| {4:<10.2f} | {5:<10.2f}")
        print(ROW_FMT.format("generator (train)", *self.train_history["generator"][-1]))
        print(ROW_FMT.format("discriminator (train)", *self.train_history["discriminator"][-1]))

        torch.save(self.generator.state_dict(), "generator_weights.pth")
        torch.save(self.discriminator.state_dict(), "discriminator_weights.pth")

        pickle.dump({"train": self.train_history}, open(self.pklfile, "wb"))
        print("train-loss:" + str(self.train_history["generator"][-1][0]))
    '''
    def configure_optimizers(self):
        lr = self.hparams.lr

        optimizer_discriminator = torch.optim.RMSprop(self.discriminator.parameters(), lr)
        optimizer_generator = torch.optim.RMSprop(self.generator.parameters(), lr)
        return [optimizer_discriminator, optimizer_generator], []


#torch.set_default_tensor_type('torch.cuda.FloatTensor')
'''
data = MyDataModule()
model = ThreeDGAN()
trainer = pl.Trainer(
    accelerator="cpu",
    devices=1,
    max_epochs=3,
)
trainer.fit(model, data)
'''

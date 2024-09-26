import multiprocessing

from encoder_worker import EncoderWorker
from decoder_worker import DecoderWorker
from unet_worker import UnetWorker

if __name__ == "__main__":
    encoder_worker = EncoderWorker()
    decoder_worker = DecoderWorker()
    unet_worker = UnetWorker()

    p1 = multiprocessing.Process(target=encoder_worker.run)
    p1.start()

    p2 = multiprocessing.Process(target=unet_worker.run)
    p2.start()

    p3 = multiprocessing.Process(target=decoder_worker.run)
    p3.start()

    p1.join()
    p2.join()
    p3.join()
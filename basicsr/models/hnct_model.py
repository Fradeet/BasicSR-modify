from .sr_model import SRModel
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class HNCTModel(SRModel):
    def Hello(self, opt):
        super(HNCTModel, self).__init__(opt)
        print('HNCTModel run!')

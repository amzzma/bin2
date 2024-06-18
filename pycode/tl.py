import struct
from collections import OrderedDict
import numpy as np
import torch
import json


class od2bin_f32:
    def __init__(self, od: OrderedDict):
        self.od = od
        self.ba = bytearray()
        self.config = OrderedDict()

    @staticmethod
    def tensor2np_arr_bf32(tensor) -> bytes:
        array = tensor.numpy().astype(np.float32)
        return array.tobytes()

    def od2bf32(self) -> bytearray:
        temp_conf = OrderedDict()
        content_byte = bytearray()
        keys = self.od.keys()
        for key in keys:
            tb = self.tensor2np_arr_bf32(self.od[key])
            assert len(tb) % 4 == 0
            ts = self.od[key].shape
            ten_head = f"<{key}|{list(ts)}>".encode("utf-8")
            ten_end = f"</{key}>".encode("utf-8")
            fb = f"{len(ten_head)}s"
            fe = f"{len(ten_end)}s"

            temp_conf[f"<{key}|{list(ts)}>"] = len(f"<{key}|{list(ts)}>")
            temp_conf[f"{key}_data"] = len(tb)
            temp_conf[f"</{key}>"] = len(f"</{key}>")

            content_byte.extend(struct.pack(fb, ten_head))
            content_byte.extend(tb)
            content_byte.extend(struct.pack(fe, ten_end))

        meta_len = len(f"<point|{len(content_byte)}></point>") + len(content_byte)
        start_str = f"<point|{meta_len}>".encode("utf-8")
        end_str = b"</point>"

        self.config[f"<point|{meta_len}>"] = len(f"<point|{meta_len}>")
        self.config.update(temp_conf)
        self.config[f"</point>"] = len("</point>")

        ba = bytearray(start_str)
        ba.extend(content_byte)
        ba.extend(end_str)
        self.ba = ba
        return ba

    def bin_f32_write2(self, path):
        with open(f"{path}.bin", mode="wb") as f:
            f.write(self.ba)
        with open(f"{path}.json", mode="w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=4)
            f.flush()


def decode_bf32(ab) -> tuple:
    assert len(ab) % 4 == 0
    seq_len = len(ab) // 4
    fs = f"{seq_len}f"
    return struct.unpack(fs, ab)


def point_bf32_to_od(ba: bytearray):  # -> OrderedDict
    pass

# It also cpmpletely convert on other model.This a example
def check_bert():
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(r"bert")
    model = AutoModelForTokenClassification.from_pretrained(r"bert")
    od = model.state_dict()
    o2 = od2bin_f32(od)
    o2.od2bf32()
    # o2.bin_f32_write2("bert")
    print(od)
    print(o2.ba)
    print(o2.config)


def main():
    x1 = torch.arange(11 * 45141).reshape(11, 45141) * 0.00003
    x2 = torch.arange(45141 * 19).reshape(45141, 19) * 0.00003
    od1 = OrderedDict([('layer_1', x1), ('bXXX', x2)])

    ba = od2bin_f32(od1)
    ba.od2bf32()
    ba.bin_f32_write2("netT")
    print(od1)
    # print(ba.ba)
    print(ba.config)


if __name__ == "__main__":
    # test()
    # check_bert()
    main()


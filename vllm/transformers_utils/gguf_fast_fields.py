"""Fast, structure-identical parsing of GGUF metadata arrays.

gguf-py builds every array element through recursive ``_get`` calls on the
memmap, which costs ~100 s of CPU per process for a 250K-token vocabulary.
This reader parses arrays of strings and scalars directly while producing the
same ``parts``/``data`` layout, dtypes and values. Nested arrays and swapped
byte order keep the upstream implementation.
"""

import gguf
import numpy as np
from gguf import GGUFValueType


class FastFieldsReader(gguf.GGUFReader):
    def _get_field_parts(self, orig_offs, raw_type):
        if raw_type != GGUFValueType.ARRAY or self.byte_order != "I":
            return super()._get_field_parts(orig_offs, raw_type)
        buf = self.data.view(np.ndarray)
        raw_itype = self._get(orig_offs, np.uint32)
        alen = self._get(orig_offs + 4, np.uint64)
        itype = GGUFValueType(int(raw_itype[0]))
        count = int(alen[0])
        offs = orig_offs + 12
        parts = [raw_itype, alen]
        types = [GGUFValueType.ARRAY]
        if itype == GGUFValueType.STRING:
            u64 = np.dtype(np.uint64).newbyteorder("I")
            u8 = np.dtype(np.uint8).newbyteorder("I")
            mv = memoryview(buf)
            for _ in range(count):
                n = int.from_bytes(mv[offs : offs + 8], "little")
                parts.append(buf[offs : offs + 8].view(u64))
                parts.append(buf[offs + 8 : offs + 8 + n].view(u8))
                offs += 8 + n
            if count:
                types.append(GGUFValueType.STRING)
            return offs - orig_offs, parts, list(range(3, 2 + 2 * count, 2)), types
        nptype = self.gguf_scalar_to_np.get(itype)
        if nptype is None:
            return super()._get_field_parts(orig_offs, raw_type)
        dtype = np.dtype(nptype).newbyteorder("I")
        values = buf[offs : offs + count * dtype.itemsize].view(dtype)
        parts.extend(values[i : i + 1] for i in range(count))
        if count:
            types.append(itype)
        return 12 + count * dtype.itemsize, parts, list(range(2, 2 + count)), types

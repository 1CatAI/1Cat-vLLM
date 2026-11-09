# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
import random
import time
import traceback
import unittest
from multiprocessing import Lock

import torch

# Assuming these are imported from your module
from vllm.distributed.device_communicators.shm_object_storage import (
    MsgpackSerde,
    SingleWriterShmObjectStorage,
    SingleWriterShmRingBuffer,
)
from vllm.multimodal.inputs import (
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalSharedField,
)


def _dummy_elem(size: int):
    return MultiModalFieldElem(
        data=torch.empty((size,), dtype=torch.int8),
        field=MultiModalSharedField(batch_size=1),
    )


def _dummy_item(size_by_key: dict[str, int]):
    return MultiModalKwargsItem(
        {key: _dummy_elem(size) for key, size in size_by_key.items()}
    )


class TestShmObjectStorageReferenceCounting(unittest.TestCase):
    """One writer and four readers, as with tensor-parallel workers."""

    def setUp(self):
        ring_buffer = SingleWriterShmRingBuffer(
            data_buffer_size=1024 * 100, create=True
        )
        self.writer = SingleWriterShmObjectStorage(
            max_object_size=1024 * 10,
            n_readers=4,
            ring_buffer=ring_buffer,
            serde_class=MsgpackSerde,
        )
        handle = self.writer.handle()
        handle.reader_lock = Lock()
        self.readers = [
            SingleWriterShmObjectStorage.create_from_handle(handle) for _ in range(4)
        ]

    def tearDown(self):
        for reader in self.readers:
            reader.close()
        self.writer.close()

    def _is_evictable(self, key: str) -> bool:
        self.writer.free_unused()
        return not self.writer.is_cached(key)

    def test_second_request_before_all_readers_fetched_the_first(self):
        """Two requests carry the same new item in one scheduler step.

        Every reader starts on the first request at the same time, and the
        readers that finish it early go on to the second request while the
        others are still fetching. With the conditional reader count this
        ordering left the item at 10 of the 12 counts it needed.
        """
        first = self.writer.put("image", _dummy_item({"pixels": 64}))
        self.writer.touch("image")
        second = self.writer.get_cached("image")
        self.writer.release_touches()
        self.assertEqual(first, second)

        for reader in self.readers:
            reader.touch("image", address=first[0], monotonic_id=first[1])
        for reader in self.readers:
            reader.get(*first)
            reader.touch("image", address=second[0], monotonic_id=second[1])
            reader.get(*second)

        self.assertTrue(self._is_evictable("image"))

    def test_item_stays_until_every_reference_is_fetched(self):
        first = self.writer.put("image", _dummy_item({"pixels": 64}))
        self.writer.touch("image")
        second = self.writer.get_cached("image")
        self.writer.release_touches()

        for reader in self.readers:
            reader.get(*first)
        self.assertFalse(self._is_evictable("image"))
        for reader in self.readers[:-1]:
            reader.get(*second)
        self.assertFalse(self._is_evictable("image"))
        self.readers[-1].get(*second)
        self.assertTrue(self._is_evictable("image"))

    def test_touch_protects_a_fetched_item_until_it_is_looked_up(self):
        address, monotonic_id = self.writer.put("image", _dummy_item({"pixels": 64}))
        for reader in self.readers:
            reader.get(address, monotonic_id)

        self.writer.touch("image")
        self.assertFalse(self._is_evictable("image"))
        self.assertEqual(self.writer.get_cached("image"), (address, monotonic_id))
        self.writer.release_touches()
        self.assertFalse(self._is_evictable("image"))
        for reader in self.readers:
            reader.get(address, monotonic_id)
        self.assertTrue(self._is_evictable("image"))

    def test_touch_without_lookup_is_released(self):
        address, monotonic_id = self.writer.put("image", _dummy_item({"pixels": 64}))
        for reader in self.readers:
            reader.get(address, monotonic_id)

        self.writer.touch("image")
        self.writer.touch("missing")
        self.writer.release_touches()
        self.assertTrue(self._is_evictable("image"))

    def test_item_twice_in_one_request(self):
        first = self.writer.put("image", _dummy_item({"pixels": 64}))
        second = self.writer.get_cached("image")
        self.writer.release_touches()

        for reader in self.readers[:-1]:
            reader.get(*first)
            reader.get(*second)
        self.readers[-1].get(*first)
        self.assertFalse(self._is_evictable("image"))
        self.readers[-1].get(*second)
        self.assertTrue(self._is_evictable("image"))

    def test_reader_touch_does_not_write_to_shared_memory(self):
        address, monotonic_id = self.writer.put("image", _dummy_item({"pixels": 64}))
        for reader in self.readers:
            reader.get(address, monotonic_id)
        before = bytes(self.writer.ring_buffer.shared_memory.buf)

        self.readers[0].touch("image", address=address, monotonic_id=monotonic_id)

        self.assertEqual(bytes(self.writer.ring_buffer.shared_memory.buf), before)


class TestSingleWriterShmObjectStorage(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        ring_buffer = SingleWriterShmRingBuffer(
            data_buffer_size=1024 * 100,
            create=True,  # 10 MB buffer
        )
        self.storage = SingleWriterShmObjectStorage(
            max_object_size=1024 * 10,  # 10KB max object
            n_readers=2,
            ring_buffer=ring_buffer,
            serde_class=MsgpackSerde,
            reader_lock=Lock(),
        )

    def tearDown(self):
        """Clean up after each test."""
        if self.storage:
            self.storage.close()

    def test_minimal_put_get_cycle(self):
        """Test basic put and get operations."""
        key = "test_key"
        value = _dummy_item({"field1": 10, "field2": 20})

        # Put operation
        address, monotonic_id = self.storage.put(key, value)

        # Verify key is in index
        self.assertIn(key, self.storage.key_index)
        self.assertEqual(self.storage.key_index[key], (address, monotonic_id))
        self.assertEqual(self.storage.id_index[monotonic_id], key)

        # Get operation
        result = self.storage.get(address, monotonic_id)

        # Verify result
        self.assertEqual(result, value)

    def test_put_same_key_twice(self):
        """Test behavior when putting the same key multiple times."""
        key = "duplicate_key"
        value1 = "first value"
        value2 = "second value"

        # First put
        address1, id1 = self.storage.put(key, value1)
        retrieved1 = self.storage.get(address1, id1)
        self.assertEqual(retrieved1, value1)

        # should raise an error on second put
        with self.assertRaises(ValueError) as context:
            self.storage.put(key, value2)

        self.assertIn("already exists in the storage", str(context.exception))

    def test_large_object_rejection(self):
        """Test that objects exceeding max_object_size are rejected."""
        # Create an object larger than max_object_size
        large_data = "x" * (self.storage.max_object_size + 100)

        with self.assertRaises(ValueError) as context:
            self.storage.put("large_key", large_data)

        self.assertIn("exceeds max object size", str(context.exception))

    def test_buffer_overflow_and_cleanup(self):
        """Test behavior when buffer fills up and needs cleanup."""
        # Fill up the buffer with many small objects
        stored_items = []

        try:
            for i in range(1000):  # Try to store many items
                key = f"item_{i}"
                value = f"data_{i}" * 100  # Make it reasonably sized
                address, monotonic_id = self.storage.put(key, value)
                stored_items.append((key, value, address, monotonic_id))
        except MemoryError:
            print(f"Buffer filled after {len(stored_items)} items")

        # Verify that some items are still accessible
        accessible_count = 0
        for key, original_value, address, monotonic_id in stored_items:
            for i in range(self.storage.n_readers):
                retrieved = self.storage.get(address, monotonic_id)
            if retrieved == original_value:
                accessible_count += 1

        self.assertEqual(accessible_count, len(stored_items))

        try:
            for i in range(len(stored_items), 1000):  # Try to store many items
                key = f"item_{i}"
                value = f"data_{i}" * 100  # Make it reasonably sized
                address, monotonic_id = self.storage.put(key, value)
                stored_items.append((key, value, address, monotonic_id))
        except MemoryError:
            print(f"Buffer filled after {len(stored_items)} items")

        # Verify that some items are still accessibles
        for key, original_value, address, monotonic_id in stored_items:
            try:
                for i in range(self.storage.n_readers):
                    retrieved = self.storage.get(address, monotonic_id)
                if retrieved == original_value:
                    accessible_count += 1
            except ValueError as e:
                print(f"Error retrieving {key}: {e}")

        # some items from the first batch may still be accessible
        self.assertGreaterEqual(accessible_count, len(stored_items))

    def test_blocking_unread_object(self):
        """Test behavior when buffer fills up and needs cleanup."""
        # Fill up the buffer with many small objects
        stored_items = []

        try:
            for i in range(1000):  # Try to store many items
                key = f"item_{i}"
                value = f"data_{i}" * 100  # Make it reasonably sized
                address, monotonic_id = self.storage.put(key, value)
                stored_items.append((key, value, address, monotonic_id))
        except MemoryError:
            print(f"Buffer filled after {len(stored_items)} items")

        # read all items except the first one
        # to simulate a blocking situation
        accessible_count = 0
        for key, original_value, address, monotonic_id in stored_items[1:]:
            for i in range(self.storage.n_readers):
                retrieved = self.storage.get(address, monotonic_id)
            if retrieved == original_value:
                accessible_count += 1

        self.assertEqual(accessible_count, len(stored_items) - 1)

        try:
            key = f"item_{len(stored_items)}"
            value = f"data_{len(stored_items)}" * 100
            address, monotonic_id = self.storage.put(key, value)
        except MemoryError:
            print(f"Buffer filled after {len(stored_items)} items")

        # read the first item
        for i in range(self.storage.n_readers):
            key, original_value, address, monotonic_id = stored_items[0]
            retrieved = self.storage.get(address, monotonic_id)
            self.assertEqual(retrieved, original_value)

        try:
            for i in range(len(stored_items), 1000):  # Try to store many items
                key = f"item_{i}"
                value = f"data_{i}" * 100  # Make it reasonably sized
                address, monotonic_id = self.storage.put(key, value)
                stored_items.append((key, value, address, monotonic_id))
        except MemoryError:
            print(f"Buffer filled after {len(stored_items)} items")

        # some items from the first batch may still be accessible
        self.assertGreaterEqual(len(stored_items), accessible_count + 10)

    def test_invalid_get_operations(self):
        """Test various invalid get operations."""
        # Test with non-existent address
        with self.assertRaises(ValueError):  # Could be various exceptions
            self.storage.get(99999, 1)

        # Store something first
        address, monotonic_id = self.storage.put("test", "value")

        # Test with wrong monotonic_id
        with self.assertRaises(ValueError) as context:
            self.storage.get(address, monotonic_id + 100)

        self.assertIn("has been modified or is invalid", str(context.exception))

    def test_clear_storage(self):
        """Test clearing the storage."""
        # Store some items
        for i in range(5):
            self.storage.put(f"item_{i}", f"value_{i}")

        # Clear the storage
        self.storage.clear()

        # Verify that all indices are empty
        self.assertEqual(len(self.storage.key_index), 0)
        self.assertEqual(len(self.storage.id_index), 0)
        self.assertEqual(len(self.storage.ring_buffer.metadata), 0)

        # Verify that new items can be added after clearing
        address, monotonic_id = self.storage.put("new_item", "new_value")
        self.assertIn("new_item", self.storage.key_index)
        self.assertEqual((address, monotonic_id), (0, 0))


# Reader process function
def reader_process(process_id, storage_handle, items_to_read):
    """Reader process that connects to existing shared memory and reads data."""
    reader_storage = SingleWriterShmObjectStorage.create_from_handle(storage_handle)

    print(f"Reader {process_id} started")

    errors = []

    for key, original_value, address, monotonic_id in items_to_read:
        time.sleep(random.random() / 100)
        try:
            # Read data from shared memory
            retrieved_value = reader_storage.get(address, monotonic_id)

            # Verify data integrity
            assert retrieved_value == original_value
            print(f"Reader {process_id} retrieved {key}: {retrieved_value}")
        except Exception as e:
            errors.append((key, str(e), type(e).__name__))


def run_multiprocess_example():
    """Run a minimal working example with real shared memory."""
    print("=== Minimal Object Storage Example ===")

    try:
        # Create storage instance
        ring_buffer = SingleWriterShmRingBuffer(
            data_buffer_size=1024 * 100,
            create=True,  # 10 MB buffer
        )
        storage = SingleWriterShmObjectStorage(
            max_object_size=1024,
            n_readers=3,
            ring_buffer=ring_buffer,
            serde_class=MsgpackSerde,
            reader_lock=Lock(),
        )

        print(f"Created storage (writer: {storage.is_writer})")

        # Test basic data types
        test_data = [
            ("user_data", {"name": "Alice", "age": 30, "scores": [95, 87, 92]}),
            ("simple_string", "Hello, World!"),
            ("number", 42),
            ("list_data", [1, 2, 3, "four", 5.0]),
        ]

        stored_items = []

        # Store all data
        for key, value in test_data:
            print(f"Storing {key}: {value}")
            address, monotonic_id = storage.put(key, value)
            stored_items.append((key, value, address, monotonic_id))
            print(f"  -> Stored at address {address}, ID {monotonic_id}")

        print("\n--- Retrieving Data ---")
        processes = []
        handle = storage.handle()
        # initialize lock for reader processes
        handle.reader_lock = Lock()
        for i in range(storage.n_readers):
            p = multiprocessing.Process(
                target=reader_process, args=(i, handle, stored_items)
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join()

    except Exception as e:
        print(f"Error in minimal example: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    # Run the minimal example first
    run_multiprocess_example()
    print("\n" + "=" * 50 + "\n")

    # Run the test suite
    print("Running comprehensive test suite...")
    unittest.main(verbosity=2, exit=False)

import csv
import json
import os

import numpy as np
import torch


class Exp_Basic_Test:
    def __init__(self):
        self._checkpoint_info = {
            'checkpoint_path': None,
            'strict_load': None,
            'missing_keys': [],
            'unexpected_keys': [],
        }

    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, 'module') else self.model

    def _extract_state_dict(self, checkpoint):
        if isinstance(checkpoint, dict):
            for key in ('state_dict', 'model_state_dict', 'model', 'net'):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return value
            if checkpoint and all(isinstance(key, str) for key in checkpoint.keys()):
                return checkpoint
        raise TypeError('Unsupported checkpoint format: expected a state_dict-like mapping.')

    def _resolve_checkpoint_path(self, setting):
        checkpoint_path = str(getattr(self.args, 'checkpoint_path', '') or '').strip()
        checkpoint_file = str(getattr(self.args, 'checkpoint_file', 'checkpoint.pth') or 'checkpoint.pth')

        if checkpoint_path:
            if os.path.isdir(checkpoint_path):
                checkpoint_path = os.path.join(checkpoint_path, checkpoint_file)
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError('Checkpoint not found: {}'.format(checkpoint_path))
            return checkpoint_path

        resolved_setting = str(getattr(self.args, 'setting', '') or setting).strip()
        if not resolved_setting:
            raise ValueError('A setting name is required when checkpoint_path is not provided.')

        checkpoint_path = os.path.join(self.args.checkpoints, resolved_setting, checkpoint_file)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError('Checkpoint not found: {}'.format(checkpoint_path))
        return checkpoint_path

    def _load_test_checkpoint(self, setting):
        if bool(getattr(self.args, 'skip_checkpoint_load', False)):
            print('Skipping checkpoint load and evaluating the current model weights.')
            return None

        checkpoint_path = self._resolve_checkpoint_path(setting)
        print('loading checkpoint from {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = self._extract_state_dict(checkpoint)

        incompatible = self._unwrap_model().load_state_dict(
            state_dict,
            strict=bool(getattr(self.args, 'strict_load', True)),
        )
        missing_keys = list(getattr(incompatible, 'missing_keys', []))
        unexpected_keys = list(getattr(incompatible, 'unexpected_keys', []))
        self._checkpoint_info = {
            'checkpoint_path': checkpoint_path,
            'strict_load': bool(getattr(self.args, 'strict_load', True)),
            'missing_keys': missing_keys,
            'unexpected_keys': unexpected_keys,
        }
        if missing_keys:
            print('missing keys: {}'.format(missing_keys))
        if unexpected_keys:
            print('unexpected keys: {}'.format(unexpected_keys))
        return checkpoint_path

    def _ensure_dir(self, path):
        os.makedirs(path, exist_ok=True)
        return path

    def _json_ready(self, value):
        if isinstance(value, dict):
            return {str(key): self._json_ready(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_ready(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _write_json(self, path, payload):
        with open(path, 'w', encoding='utf-8') as file_obj:
            json.dump(self._json_ready(payload), file_obj, indent=2, ensure_ascii=False)

    def _write_csv_rows(self, path, rows):
        rows = list(rows)
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with open(path, 'w', encoding='utf-8', newline='') as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: self._json_ready(value) for key, value in row.items()})

    def _write_npz(self, path, **arrays):
        np.savez_compressed(path, **arrays)

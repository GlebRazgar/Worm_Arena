"""
Configuration loader for Worm Arena

Worm Arena uses a YAML-based configuration system for easy experimentation and modular design. The config system eliminates hardcoded values and makes it simple to switch between different connectome datasets.
"""


import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Simple configuration manager"""
    
    def __init__(self, config_dir: str = None):
        if config_dir is None:
            # Auto-detect config directory
            config_dir = Path(__file__).parent
        self.config_dir = Path(config_dir)
        
        # Load all configs
        self.data = self._load_yaml('data.yaml')
        self.model = self._load_yaml('model.yaml')
        self.train = self._load_yaml('train.yaml')
    
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a YAML config file"""
        filepath = self.config_dir / filename
        if not filepath.exists():
            return {}
        
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f) or {}
        return config
    
    @property
    def connectome(self) -> str:
        """Get the configured connectome dataset"""
        return self.data.get('dataset', 'cook2019')
    
    @property
    def functional_sources(self):
        """Get list of functional data source datasets"""
        return self.data.get('functional', {}).get('source_datasets', [])
    
    @property
    def functional_worms(self):
        """Get list of worms to load (None = all worms)"""
        return self.data.get('functional', {}).get('worms', None)
    
    @property
    def match_connectome(self):
        """Whether to filter neurons by connectome"""
        return self.data.get('functional', {}).get('match_connectome', True)
    
    def __repr__(self):
        return f"Config(connectome={self.connectome}, functional_sources={len(self.functional_sources)})"


# Singleton instance
_config = None

def get_config(config_dir: str = None) -> Config:
    """Get or create config instance"""
    global _config
    if _config is None or config_dir is not None:
        _config = Config(config_dir)
    return _config


if __name__ == "__main__":
    # Test config loading
    cfg = get_config()
    print(f"Loaded config: {cfg}")
    print(f"Connectome: {cfg.connectome}")
    print(f"Data config: {cfg.data}")
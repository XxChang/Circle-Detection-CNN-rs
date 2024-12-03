use std::{fs::File, path::{Path, PathBuf}};

use burn::data::dataset::{transform::{Mapper, MapperDataset}, Dataset, InMemDataset};
use ndarray::Array2;
use ndarray_npy::ReadNpyExt;
use serde::{Deserialize, Serialize};


#[derive(Debug, Clone, Serialize, Deserialize)]
struct CircleCSV {
    #[serde(rename = "NAME")]
    name: String,
    #[serde(rename = "ROW")]
    row: f32,
    #[serde(rename = "COL")]
    col: f32,
    #[serde(rename = "RAD")]
    rad: f32,
}

#[derive(Debug, Clone)]
pub struct CircleDatasetItem {
    pub image: [[f32;200];200],
    pub annotation: (f32, f32, f32),
}

struct CircleDatasetMapperUtils {
    pub parent_path: PathBuf
}

impl Mapper<CircleCSV, CircleDatasetItem> for CircleDatasetMapperUtils {
    fn map(&self, item: &CircleCSV) -> CircleDatasetItem {
        let npy_path = self.parent_path
            .join(item.name.as_str());
        let reader = File::open(npy_path).unwrap();
        let arr = Array2::<f64>::read_npy(reader).unwrap();
        let mut image = [[0.0;200];200];
        for ((i, j), &value) in arr.indexed_iter() {
            image[i][j] = value as f32;
        }
        
        CircleDatasetItem {
            image,
            annotation: (item.row, item.col, item.rad),
        }
    }
}

type CircleDatasetMapper = MapperDataset<InMemDataset<CircleCSV>, CircleDatasetMapperUtils, CircleCSV>;

pub struct CircleDataset {
    dataset: CircleDatasetMapper,
}

impl Dataset<CircleDatasetItem> for CircleDataset {
    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn get(&self, index: usize) -> Option<CircleDatasetItem> {
        self.dataset.get(index)
    }
}

impl CircleDataset {
    pub fn read_train_csv(path: &str) -> CircleDataset {
        let mut rdr = csv::ReaderBuilder::new();
        rdr.delimiter(b',');
        let dataset: InMemDataset<CircleCSV> = InMemDataset::from_csv(path, &rdr).unwrap();

        let mapper = CircleDatasetMapperUtils {
            parent_path: Path::new(path).parent().unwrap().to_path_buf()
        };
        let dataset = MapperDataset::new(dataset, mapper);

        CircleDataset { dataset }
    }
}
